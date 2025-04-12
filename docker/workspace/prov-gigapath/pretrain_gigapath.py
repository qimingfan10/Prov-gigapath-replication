#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import random
import gigapath
from gigapath.pipeline import tile_one_slide, load_tile_encoder_transforms
from gigapath.slide_encoder import LongNetViT


class TileDataset(Dataset):
    """用于预训练的切片数据集"""
    def __init__(self, image_paths, transform=None, mask_ratio=0.75):
        self.image_paths = image_paths
        self.transform = transform
        self.mask_ratio = mask_ratio
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 加载图像
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        
        return img


class MaskedAutoencoder(nn.Module):
    """
    基于MAE（Masked Autoencoder）的自监督学习模型
    参考: https://arxiv.org/abs/2111.06377
    """
    def __init__(self, encoder, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder  # 使用GigaPath的tile encoder作为编码器
        self.mask_ratio = mask_ratio
        
        # 解码器（简化版）
        self.decoder = nn.Sequential(
            nn.Linear(1536, decoder_dim),  # GigaPath tile encoder输出维度为1536
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 3 * 224 * 224)  # 重建为RGB图像
        )
    
    def random_masking(self, x, mask_ratio):
        """随机掩码"""
        N, C, H, W = x.shape
        L = H * W
        len_keep = int(L * (1 - mask_ratio))
        
        # 生成随机掩码
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留的tokens
        ids_keep = ids_shuffle[:, :len_keep]
        
        # 掩码后的图像
        x_masked = torch.zeros_like(x)
        for i in range(N):
            # 将图像展平为tokens
            tokens = x[i].reshape(C, -1).transpose(0, 1)  # L, C
            # 只保留未掩码的tokens
            tokens_keep = tokens[ids_keep[i]]
            # 重建为图像形状
            tokens_masked = torch.zeros_like(tokens)
            tokens_masked[ids_keep[i]] = tokens_keep
            x_masked[i] = tokens_masked.transpose(0, 1).reshape(C, H, W)
        
        return x_masked, ids_restore
    
    def forward(self, imgs):
        # 随机掩码
        imgs_masked, ids_restore = self.random_masking(imgs, self.mask_ratio)
        
        # 编码
        latent = self.encoder(imgs_masked)
        
        # 解码并重建
        pred = self.decoder(latent)
        pred = pred.reshape(pred.shape[0], 3, 224, 224)
        
        # 计算重建损失（只在掩码区域）
        loss = F.mse_loss(pred, imgs, reduction='mean')
        
        return loss, pred


def collect_image_paths(data_dir, extensions=('.png', '.jpg', '.jpeg')):
    """收集所有图像路径"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True))
    return image_paths


def pretrain_tile_encoder(image_paths, output_dir, batch_size=64, num_epochs=100, learning_rate=1e-4, mask_ratio=0.75, use_gpu=True):
    """预训练tile encoder"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 减小批量大小以适应内存
    if not use_gpu:
        #batch_size = min(batch_size, 16)  # CPU模式下使用更小的批量
        print(f"使用批量大小: {batch_size}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    dataset = TileDataset(image_paths, transform=transform, mask_ratio=mask_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 创建模型
    # 使用timm加载模型架构，但不加载预训练权重
    encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
    model = MaskedAutoencoder(encoder, mask_ratio=mask_ratio)
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device)
            
            # 前向传播
            loss, _ = model(imgs)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新学习率（在 optimizer.step() 之后）
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(output_dir, 'best_tile_encoder.pth'))
            print(f"保存最佳模型，损失: {best_loss:.6f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f'tile_encoder_epoch_{epoch+1}.pth'))
    
    print(f"预训练完成！最佳损失: {best_loss:.6f}")
    return os.path.join(output_dir, 'best_tile_encoder.pth')


def pretrain_slide_encoder(tile_encoder_path, image_dirs, output_dir, batch_size=16, num_epochs=50, learning_rate=1e-4, use_gpu=False):
    """预训练slide encoder"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 减小批量大小以适应内存
    if not use_gpu:
        #batch_size = min(batch_size, 8)  # CPU模式下使用更小的批量
        print(f"使用批量大小: {batch_size}")
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练的tile encoder
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
    checkpoint = torch.load(tile_encoder_path, map_location='cpu')
    tile_encoder.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建一个简化版的slide encoder，避免使用Flash Attention
    class SimpleSlideEncoder(nn.Module):
        def __init__(self, in_dim=1536, hidden_dim=768, out_dim=768):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, out_dim)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(hidden_dim)
            
        def forward(self, x, coords=None):
            # x: [B, N, D]
            # 简单地对所有特征进行平均池化
            x = x.mean(dim=1)  # [B, D]
            x = self.fc1(x)
            x = self.act(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.act(x)
            x = self.norm(x)
            x = self.fc3(x)
            return x
    
    # 使用简化版的slide encoder
    print("使用slide encoder")
    slide_encoder = SimpleSlideEncoder(in_dim=1536, hidden_dim=768, out_dim=768)
    
    # 将模型移至设备
    tile_encoder = tile_encoder.to(device)
    slide_encoder = slide_encoder.to(device)
    
    # 设置为评估模式，不更新参数
    tile_encoder.eval()
    
    # 定义优化器
    optimizer = AdamW(slide_encoder.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 实现对比学习损失
    def contrastive_loss(features):
        # 改进的对比学习损失
        batch_size = features.shape[0]
        if batch_size <= 1:
            # 如果只有一个样本，返回一个小的损失值
            return torch.tensor(0.1, device=features.device, requires_grad=True)
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 创建标签：对角线上的元素是正样本（自己与自己的相似度）
        labels = torch.arange(batch_size, device=features.device)
        
        # 使用交叉熵损失，将相似度矩阵视为logits
        # 温度系数用于控制分布的平滑度
        temperature = 0.07
        loss = F.cross_entropy(similarity_matrix / temperature, labels)
        
        return loss
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        slide_encoder.train()
        epoch_loss = 0.0
        
        # 对每个slide目录进行处理
        all_slide_features = []
        for slide_dir in tqdm(image_dirs, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 获取slide中的所有图像
            image_paths = glob.glob(os.path.join(slide_dir, "**/*.png"), recursive=True)
            if not image_paths:
                continue
            
            
            
            # 过滤图像路径，只保留符合坐标格式的文件名
            valid_image_paths = []
            for path in image_paths:
                filename = os.path.basename(path)
                # 检查文件名是否包含坐标格式
                if 'x_' in filename and 'y.' in filename:
                    valid_image_paths.append(path)
                else:
                    print(f"警告：跳过不符合坐标格式的文件: {filename}")
            
            if not valid_image_paths:
                print(f"切片 {os.path.basename(slide_dir)} 没有找到符合坐标格式的图像文件")
                continue
            #############################################################
            #valid_image_paths = valid_image_paths[:2]
            #############################################################
         
            print(f"找到 {len(valid_image_paths)} 个有效图像文件用于切片 {os.path.basename(slide_dir)}")
            
               
            # 提取特征
            tile_features = []
            coords = []
            
            transform = load_tile_encoder_transforms()
            
            for img_path in valid_image_paths:
                try:
                    # 从文件名提取坐标
                    img_name = os.path.basename(img_path)
                    # 确保文件名符合格式
                    if 'x_' in img_name and 'y.' in img_name:
                        x_part, y_part = img_name.split('.png')[0].split('_')
                        x = int(x_part.replace('x', ''))
                        y = int(y_part.replace('y', ''))
                        
                        # 加载图像
                        with open(img_path, "rb") as f:
                            img = Image.open(f).convert("RGB")
                            img = transform(img)
                        
                        # 提取特征
                        with torch.no_grad():
                            img = img.unsqueeze(0).to(device)
                            feature = tile_encoder(img)
                            tile_features.append(feature.cpu())
                            coords.append(torch.tensor([x, y], dtype=torch.float))
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")
                    continue
            
            if not tile_features:
                print(f"切片 {os.path.basename(slide_dir)} 没有有效的特征")
                continue
            
            # 将特征和坐标转换为张量
            tile_features = torch.cat(tile_features, dim=0)
            coords = torch.stack(coords, dim=0)
            
            # 将特征和坐标移至GPU并转换为fp16
            if use_gpu:
                tile_features = tile_features.to(device).half()  # 转换为fp16
                coords = coords.to(device).half()  # 转换为fp16
            else:
                tile_features = tile_features.to(device)
                coords = coords.to(device)
            
            # 检查并修复维度
            if len(tile_features.shape) == 2:
                # 如果是二维张量 (N, D)，添加一个批次维度变成 (1, N, D)
                tile_features = tile_features.unsqueeze(0)
                
                # 同样处理坐标
                if len(coords.shape) == 2:
                    coords = coords.unsqueeze(0)
            
            # 打印形状以便调试
            print(f"tile_features shape: {tile_features.shape}, dtype: {tile_features.dtype}")
            print(f"coords shape: {coords.shape}, dtype: {coords.dtype}")
            
            try:
                # 前向传播
                slide_features = slide_encoder(tile_features, coords)
                
                # 如果slide_features是元组，取第一个元素
                if isinstance(slide_features, tuple):
                    slide_features = slide_features[0]
                
                # 确保slide_features是二维的 (B, D)
                if len(slide_features.shape) > 2:
                    slide_features = slide_features.squeeze(1)  # 移除序列维度
                
                # 将特征添加到列表中，稍后一起处理
                if slide_features is not None:
                    all_slide_features.append(slide_features)
            except Exception as e:
                print(f"处理slide时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 如果收集到了足够多的特征，一起进行对比学习
        if len(all_slide_features) > 1:
            all_slide_features = torch.cat(all_slide_features, dim=0)
            
            # 确保有足够的样本
            if all_slide_features.shape[0] > 1:
                loss = contrastive_loss(all_slide_features)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新学习率（在 optimizer.step() 之后）
                scheduler.step()
                
                epoch_loss = loss.item()
            else:
                print("警告：没有足够的样本进行对比学习")
                epoch_loss = 0.0
        else:
            print("警告：没有收集到有效的特征")
            epoch_loss = 0.0
        
        # 计算平均损失
        if len(image_dirs) > 0:
            avg_loss = epoch_loss / len(image_dirs)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slide_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(output_dir, 'best_slide_encoder.pth'))
                print(f"保存最佳模型，损失: {best_loss:.6f}")
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slide_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(output_dir, f'slide_encoder_epoch_{epoch+1}.pth'))
    
    print(f"预训练完成！最佳损失: {best_loss:.6f}")
    return os.path.join(output_dir, 'best_slide_encoder.pth')


def rename_wsi_files(data_dir):
    """重命名下载的WSI文件，去除'?download=1'后缀"""
    files = glob.glob(os.path.join(data_dir, "*.ndpi?*")) + glob.glob(os.path.join(data_dir, "*.svs?*"))
    renamed_files = []
    
    for file_path in files:
        new_path = file_path.split('?')[0]
        if file_path != new_path:
            os.rename(file_path, new_path)
            print(f"重命名文件: {file_path} -> {new_path}")
        renamed_files.append(new_path)
    
    # 如果没有找到带问号的文件，则直接获取所有WSI文件
    if not renamed_files:
        renamed_files = glob.glob(os.path.join(data_dir, "*.ndpi")) + glob.glob(os.path.join(data_dir, "*.svs"))
        
    return renamed_files


def preprocess_slides(slide_files, output_dir, level=0, tile_size=256):
    """预处理所有切片，将它们分割成小块"""
    os.makedirs(output_dir, exist_ok=True)
    
    processed_slides = []
    for slide_file in tqdm(slide_files, desc="处理切片"):
        # 支持 .ndpi 和 .svs 格式
        slide_id = os.path.basename(slide_file).replace('.ndpi', '').replace('.svs', '')
        slide_output_dir = os.path.join(output_dir, slide_id)
        
        # 如果已经处理过，跳过
        if os.path.exists(slide_output_dir) and len(os.listdir(slide_output_dir)) > 0:
            print(f"切片 {slide_id} 已经处理过，跳过")
            processed_slides.append(slide_output_dir)
            continue
        
        try:
            tile_one_slide(
                slide_file=slide_file,
                save_dir=slide_output_dir,
                level=level,
                tile_size=tile_size
            )
            processed_slides.append(slide_output_dir)
        except Exception as e:
            print(f"处理切片 {slide_id} 时出错: {e}")
    
    return processed_slides


def main():
    parser = argparse.ArgumentParser(description='GigaPath模型预训练流程')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/prov-gigapath/data',
                        help='包含ndpi文件的数据目录')
    parser.add_argument('--output_dir', type=str, default='/root/gigapath_pretrain',
                        help='输出目录')
    parser.add_argument('--level', type=int, default=0,
                        help='用于切片的放大级别（0为最高放大倍数）')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='切片大小')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='预训练的批量大小')
    parser.add_argument('--num_epochs_tile', type=int, default=100,
                        help='tile encoder预训练轮数')
    parser.add_argument('--num_epochs_slide', type=int, default=50,
                        help='slide encoder预训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='MAE掩码比例')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='跳过预处理步骤')
    parser.add_argument('--skip_tile_pretrain', action='store_true',
                        help='跳过tile encoder预训练')
    parser.add_argument('--skip_slide_pretrain', action='store_true',
                        help='跳过slide encoder预训练')
    parser.add_argument('--tile_encoder_path', type=str, default='',
                        help='预训练的tile encoder路径（如果跳过tile encoder预训练）')
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用GPU进行训练')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    tiles_dir = os.path.join(args.output_dir, 'tiles')
    tile_model_dir = os.path.join(args.output_dir, 'tile_model')
    slide_model_dir = os.path.join(args.output_dir, 'slide_model')
    
    # 步骤1：重命名WSI文件
    slide_files = rename_wsi_files(args.data_dir)
    
    if not slide_files:
        print(f"错误：在 {args.data_dir} 中没有找到ndpi或svs文件")
        return
    
    # 步骤2：预处理切片
    if not args.skip_preprocessing:
        processed_slides = preprocess_slides(slide_files, tiles_dir, args.level, args.tile_size)
    else:
        # 如果跳过预处理，假设切片已经处理好
        processed_slides = [os.path.join(tiles_dir, os.path.basename(f).replace('.ndpi', '').replace('.svs', '')) 
                           for f in slide_files]
    
    # 步骤3：预训练tile encoder
    if not args.skip_tile_pretrain:
        # 收集所有图像路径
        all_image_paths = []
        for slide_dir in processed_slides:
            image_paths = glob.glob(os.path.join(slide_dir, "**/*.png"), recursive=True)
            all_image_paths.extend(image_paths)
        if not all_image_paths:
            print("错误：没有找到图像文件，无法预训练tile encoder")
            return
        #############################################################
        #all_image_paths = all_image_paths[:2]
        print(f"使用 {len(all_image_paths)} 张图像预训练tile encoder")
        
        # 创建tile model目录
        os.makedirs(tile_model_dir, exist_ok=True)
        
        # 预训练tile encoder
        print("开始预训练tile encoder...")
        tile_encoder_path = pretrain_tile_encoder(
            all_image_paths,
            tile_model_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs_tile,
            learning_rate=args.learning_rate,
            mask_ratio=args.mask_ratio,
            use_gpu=args.use_gpu
        )
    else:
        # 使用已有的tile encoder权重
        if args.tile_encoder_path:
            tile_encoder_path = args.tile_encoder_path
        else:
            tile_encoder_path = '/root/autodl-fs/prov-gigapath/tile_model/best_tile_encoder.pth'
        
        print(f"使用已有的tile encoder权重: {tile_encoder_path}")
        
        # 检查文件是否存在
        if not os.path.exists(tile_encoder_path):
            print(f"错误：tile encoder权重文件不存在: {tile_encoder_path}")
            return
    
    # 步骤4：预训练slide encoder
    if not args.skip_slide_pretrain:
        os.makedirs(slide_model_dir, exist_ok=True)
        print("开始预训练slide encoder...")
        
        slide_encoder_path = pretrain_slide_encoder(
            tile_encoder_path,
            processed_slides,
            slide_model_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs_slide,
            learning_rate=args.learning_rate,
            use_gpu=args.use_gpu
        )
    else:
        slide_encoder_path = os.path.join(slide_model_dir, 'best_slide_encoder.pth')
        print(f"跳过slide encoder预训练，使用已有权重: {slide_encoder_path}")
    
    print("预训练完成！")
    print(f"Tile Encoder保存在: {tile_encoder_path}")
    print(f"Slide Encoder保存在: {slide_encoder_path}")
    
    # 验证模型文件是否存在
    print(f"Tile encoder 文件存在: {os.path.exists(tile_encoder_path)}")
    print(f"Slide encoder 文件存在: {os.path.exists(slide_encoder_path)}")
    
    # 尝试加载模型以验证
    try:
        # 尝试加载 tile encoder
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        tile_checkpoint = torch.load(tile_encoder_path, map_location='cpu')
        tile_encoder.load_state_dict(tile_checkpoint['model_state_dict'])
        print("成功加载 tile encoder")
        
        # 尝试加载 slide encoder (使用SimpleSlideEncoder而不是LongNetViT)
        class SimpleSlideEncoder(nn.Module):
            def __init__(self, in_dim=1536, hidden_dim=768, out_dim=768):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, out_dim)
                self.act = nn.GELU()
                self.norm = nn.LayerNorm(hidden_dim)
                
            def forward(self, x, coords=None):
                # x: [B, N, D]
                # 简单地对所有特征进行平均池化
                x = x.mean(dim=1)  # [B, D]
                x = self.fc1(x)
                x = self.act(x)
                x = self.norm(x)
                x = self.fc2(x)
                x = self.act(x)
                x = self.norm(x)
                x = self.fc3(x)
                return x
        
        slide_encoder = SimpleSlideEncoder(in_dim=1536, hidden_dim=768, out_dim=768)
        slide_checkpoint = torch.load(slide_encoder_path, map_location='cpu')
        slide_encoder.load_state_dict(slide_checkpoint['model_state_dict'])
        print("成功加载 slide encoder")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()