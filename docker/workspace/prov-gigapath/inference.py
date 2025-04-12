#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from gigapath.classification_head import get_model
from train_gigapath import SlideDataset


def load_model(model_path, input_dim=1536, latent_dim=768, feat_layer="11", n_classes=2):
    """加载训练好的模型"""
    model = get_model(
        input_dim=input_dim,
        latent_dim=latent_dim,
        feat_layer=feat_layer,
        n_classes=n_classes
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model


def run_inference(model, feature_dir, output_file, batch_size=16):
    """使用训练好的模型进行推理"""
    # 获取所有特征文件
    feature_files = glob.glob(os.path.join(feature_dir, "*_features.pt"))
    
    if not feature_files:
        print(f"在 {feature_dir} 中没有找到特征文件")
        return
    
    # 创建数据集和数据加载器
    dataset = SlideDataset(feature_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 进行推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    with torch.no_grad():
        for slide_embeds, slide_ids in tqdm(dataloader, desc="推理中"):
            slide_embeds = slide_embeds.to(device)
            outputs = model(slide_embeds, None)
            
            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # 保存结果
            for i, slide_id in enumerate(slide_ids):
                results.append({
                    'slide_id': slide_id,
                    'predicted_label': preds[i].item(),
                    'confidence': probs[i, preds[i]].item()
                })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"推理结果已保存到 {output_file}")
    
    # 打印结果统计
    print(f"预测标签分布: {results_df['predicted_label'].value_counts().to_dict()}")
    print(f"平均置信度: {results_df['confidence'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description='GigaPath模型推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='包含特征文件的目录')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                        help='输出预测结果的文件路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='推理的批量大小')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数量')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model_path, n_classes=args.num_classes)
    
    # 运行推理
    run_inference(model, args.feature_dir, args.output_file, args.batch_size)


if __name__ == '__main__':
    main()