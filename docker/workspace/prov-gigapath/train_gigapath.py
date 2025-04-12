#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import timm
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
import gigapath
from gigapath.pipeline import tile_one_slide, load_tile_encoder_transforms, run_inference_with_tile_encoder, run_inference_with_slide_encoder
from gigapath.classification_head import ClassificationHead, get_model
import torch.nn as nn


def rename_ndpi_files(data_dir):
    """重命名下载的ndpi文件，去除'?download=1'后缀"""
    files = glob.glob(os.path.join(data_dir, "*.ndpi?*"))
    renamed_files = []
    
    for file_path in files:
        new_path = file_path.split('?')[0]
        if file_path != new_path:
            os.rename(file_path, new_path)
            print(f"重命名文件: {file_path} -> {new_path}")
        renamed_files.append(new_path)
    
    # 如果没有找到带问号的文件，则直接获取所有ndpi文件
    if not renamed_files:
        renamed_files = glob.glob(os.path.join(data_dir, "*.ndpi"))
        
    return renamed_files


def preprocess_slides(slide_files, output_dir, level=0, tile_size=256):
    """预处理所有切片，将它们分割成小块"""
    os.makedirs(output_dir, exist_ok=True)
    
    processed_slides = []
    for slide_file in tqdm(slide_files, desc="处理切片"):
        slide_id = os.path.basename(slide_file).replace('.ndpi', '')
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


def extract_features(processed_slides, output_dir, batch_size=128, use_pretrained=True):
    """使用GigaPath模型提取特征"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 始终使用预训练模型，但可以选择是否冻结
    if not use_pretrained:
        print("警告：从头训练GigaPath模型需要大量数据和计算资源")
        print("强制使用预训练模型，但在后续训练中可以选择不冻结参数进行微调")
    
    # 加载预训练模型
    try:
        # 尝试使用标准加载方式
        tile_encoder, slide_encoder_model = gigapath.pipeline.load_tile_slide_encoder()
    except Exception as e:
        print(f"标准加载失败，尝试加载简化版slide encoder: {e}")
        
        # 加载tile encoder
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        
        # 创建简化版slide encoder
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
        
        # 加载预训练的slide encoder
        slide_encoder_model = SimpleSlideEncoder(in_dim=1536, hidden_dim=768, out_dim=768)
        slide_encoder_path = os.path.join('/root/autodl-fs/prov-gigapath/slide_model', 'best_slide_encoder.pth')
        if os.path.exists(slide_encoder_path):
            slide_checkpoint = torch.load(slide_encoder_path, map_location='cpu')
            slide_encoder_model.load_state_dict(slide_checkpoint['model_state_dict'])
            print(f"成功加载简化版slide encoder: {slide_encoder_path}")
        else:
            print(f"警告：找不到预训练的slide encoder: {slide_encoder_path}")
    
    for slide_dir in tqdm(processed_slides, desc="提取特征"):
        slide_id = os.path.basename(slide_dir)
        feature_output_path = os.path.join(output_dir, f"{slide_id}_features.pt")
        
        # 如果已经提取过特征，跳过
        if os.path.exists(feature_output_path):
            print(f"切片 {slide_id} 的特征已经提取过，跳过")
            continue
        
        # 直接搜索PNG文件，这是tile_one_slide函数生成的文件格式
        image_paths = []
        for ext in ['.png']:  # 只搜索PNG文件，因为这是tile_one_slide生成的格式
            found_files = glob.glob(os.path.join(slide_dir, f"**/*{ext}"), recursive=True)
            image_paths.extend(found_files)
        
        # 过滤图像路径，只保留符合坐标格式的文件名（例如：04352x_39168y.png）
        valid_image_paths = []
        for path in image_paths:
            filename = os.path.basename(path)
            # 检查文件名是否包含坐标格式
            if 'x_' in filename and 'y.' in filename:
                valid_image_paths.append(path)
            else:
                print(f"警告：跳过不符合坐标格式的文件: {filename}")
        
        if not valid_image_paths:
            print(f"切片 {slide_id} 没有找到符合坐标格式的图像文件")
            continue
        
        print(f"找到 {len(valid_image_paths)} 个有效图像文件用于切片 {slide_id}")
        
        # 提取特征
        try:
            # 提取切片特征
            tile_outputs = run_inference_with_tile_encoder(valid_image_paths, tile_encoder, batch_size=batch_size)
            
            # 提取滑片特征
            slide_outputs = run_inference_with_slide_encoder(
                tile_outputs['tile_embeds'], 
                tile_outputs['coords'], 
                slide_encoder_model
            )
            
            # 保存特征
            torch.save({
                'tile_embeds': tile_outputs['tile_embeds'],
                'coords': tile_outputs['coords'],
                'slide_embeds': slide_outputs['last_layer_embed']
            }, feature_output_path)
            
            print(f"成功提取并保存切片 {slide_id} 的特征")
        except Exception as e:
            print(f"提取切片 {slide_id} 特征时出错: {e}")
            import traceback
            traceback.print_exc()


class SlideDataset(Dataset):
    """滑片特征数据集"""
    def __init__(self, feature_files, labels=None):
        self.feature_files = feature_files
        self.labels = labels  # 可以是None（用于推理）或者是一个字典 {slide_id: label}
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        slide_id = os.path.basename(feature_file).replace('_features.pt', '')
        
        # 加载特征
        data = torch.load(feature_file)
        slide_embed = data['slide_embeds']
        
        if self.labels is not None and slide_id in self.labels:
            label = self.labels[slide_id]
            return slide_embed, torch.tensor(label, dtype=torch.long)
        else:
            return slide_embed, slide_id  # 用于推理


def train_model(feature_dir, labels_file, output_dir, batch_size=16, num_epochs=50, learning_rate=1e-4, freeze_pretrained=True):
    """训练GigaPath分类模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查标签文件是否存在
    if not os.path.exists(labels_file):
        print(f"错误：标签文件 {labels_file} 不存在")
        return
    
    # 加载标签
    labels_df = pd.read_csv(labels_file)
    print(f"加载标签文件: {labels_file}")
    print(f"标签文件包含 {len(labels_df)} 个样本")
    
    # 检查标签文件格式
    if 'slide_id' not in labels_df.columns or 'label' not in labels_df.columns:
        print(f"错误：标签文件格式不正确，需要包含 'slide_id' 和 'label' 列")
        print(f"当前列: {labels_df.columns.tolist()}")
        return
    
    labels = {row['slide_id']: row['label'] for _, row in labels_df.iterrows()}
    
    # 获取所有特征文件
    feature_files = glob.glob(os.path.join(feature_dir, "*_features.pt"))
    print(f"找到 {len(feature_files)} 个特征文件")
    
    # 只保留有标签的特征文件
    labeled_feature_files = []
    for f in feature_files:
        slide_id = os.path.basename(f).replace('_features.pt', '')
        if slide_id in labels:
            labeled_feature_files.append(f)
        else:
            print(f"警告：特征文件 {f} 没有对应的标签")
    
    if not labeled_feature_files:
        print("没有找到带标签的特征文件，无法训练模型")
        print("特征文件 slide_id:")
        for f in feature_files:
            print(f"  - {os.path.basename(f).replace('_features.pt', '')}")
        print("标签文件 slide_id:")
        for slide_id in labels.keys():
            print(f"  - {slide_id}")
        return
    
    print(f"找到 {len(labeled_feature_files)} 个带标签的特征文件")
    
    # 划分训练集和验证集
    train_files, val_files = train_test_split(labeled_feature_files, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = SlideDataset(train_files, labels)
    val_dataset = SlideDataset(val_files, labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    num_classes = len(set(labels.values()))
    model = get_model(
        input_dim=1536,  # GigaPath tile encoder输出维度
        latent_dim=768,  # GigaPath slide encoder输出维度
        feat_layer="11",  # 使用最后一层特征
        n_classes=num_classes,
        freeze=freeze_pretrained  # 是否冻结预训练模型
    )
    
    # 将模型移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for slide_embeds, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            slide_embeds = slide_embeds.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(slide_embeds, None)  # 在训练时不需要坐标，因为我们直接使用提取的特征
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for slide_embeds, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                slide_embeds = slide_embeds.to(device)
                labels = labels.to(device)
                
                outputs = model(slide_embeds, None)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 打印统计信息
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")


def create_dummy_labels(feature_dir, output_file, num_classes=2):
    """创建虚拟标签文件用于测试（如果没有真实标签）"""
    feature_files = glob.glob(os.path.join(feature_dir, "*_features.pt"))
    
    if not feature_files:
        print(f"警告：在 {feature_dir} 中没有找到特征文件，无法创建标签")
        return None
    
    # 从特征文件名中提取slide_id
    slide_ids = [os.path.basename(f).replace('_features.pt', '') for f in feature_files]
    
    # 随机分配标签
    np.random.seed(42)
    labels = np.random.randint(0, num_classes, size=len(slide_ids))
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'label': labels
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"创建虚拟标签文件: {output_file}")
    print(f"标签分布: {pd.Series(labels).value_counts().to_dict()}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='GigaPath模型训练流程')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/prov-gigapath/data',
                        help='包含ndpi文件的数据目录')
    parser.add_argument('--output_dir', type=str, default='/root/gigapath_output',
                        help='输出目录')
    parser.add_argument('--level', type=int, default=0,
                        help='用于切片的放大级别（0为最高放大倍数）')
    parser.add_argument('--tile_size', type=int, default=256,
                        help='切片大小')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='特征提取的批量大小')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='训练的批量大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--labels_file', type=str, default='',
                        help='标签文件路径（如果没有，将创建虚拟标签）')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数量')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='跳过预处理步骤')
    parser.add_argument('--skip_feature_extraction', action='store_true',
                        help='跳过特征提取步骤')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                        help='使用预训练的GigaPath模型（推荐）')
    parser.add_argument('--no_pretrained', dest='use_pretrained', action='store_false',
                        help='不使用预训练的GigaPath模型（不推荐）')
    parser.add_argument('--freeze_pretrained', action='store_true', default=True,
                        help='冻结预训练模型的参数（推荐）')
    parser.add_argument('--no_freeze_pretrained', dest='freeze_pretrained', action='store_false',
                        help='不冻结预训练模型的参数（用于微调整个模型）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    tiles_dir = os.path.join(args.output_dir, 'tiles')
    features_dir = os.path.join(args.output_dir, 'features')
    model_dir = os.path.join(args.output_dir, 'model')
    
    # 步骤1：重命名ndpi文件
    slide_files = rename_ndpi_files(args.data_dir)
    
    if not slide_files:
        print(f"错误：在 {args.data_dir} 中没有找到ndpi文件")
        return
    
    # 步骤2：预处理切片
    if not args.skip_preprocessing:
        processed_slides = preprocess_slides(slide_files, tiles_dir, args.level, args.tile_size)
    else:
        # 如果跳过预处理，假设切片已经处理好
        processed_slides = [os.path.join(tiles_dir, os.path.basename(f).replace('.ndpi', '')) 
                           for f in slide_files]
    
    # 步骤3：提取特征
    if not args.skip_feature_extraction:
        extract_features(processed_slides, features_dir, args.batch_size, use_pretrained=args.use_pretrained)
    
    # 步骤4：准备标签
    if not args.labels_file:
        args.labels_file = create_dummy_labels(features_dir, 
                                              os.path.join(args.output_dir, 'dummy_labels.csv'),
                                              args.num_classes)
        
        if not args.labels_file or not os.path.exists(args.labels_file):
            print("错误：无法创建标签文件，训练终止")
            return
    
    # 步骤5：训练模型
    train_model(features_dir, args.labels_file, model_dir, 
               args.train_batch_size, args.num_epochs, args.learning_rate, 
               freeze_pretrained=args.freeze_pretrained)


if __name__ == '__main__':
    main()