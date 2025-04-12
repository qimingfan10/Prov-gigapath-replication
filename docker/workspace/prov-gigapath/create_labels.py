#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def create_labels_file(data_dir, output_file, num_classes=2):
    """
    为数据集创建标签文件
    
    参数:
    data_dir: 包含ndpi文件的目录
    output_file: 输出标签文件的路径
    num_classes: 分类类别数量
    """
    # 获取所有ndpi文件
    ndpi_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.ndpi') or '?' in file and file.split('?')[0].endswith('.ndpi'):
            slide_id = file.split('.ndpi')[0]
            if '?' in slide_id:
                slide_id = slide_id.split('?')[0]
            ndpi_files.append(slide_id)
    
    # 随机分配标签
    np.random.seed(42)
    labels = np.random.randint(0, num_classes, size=len(ndpi_files))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'slide_id': ndpi_files,
        'label': labels
    })
    
    # 保存标签文件
    df.to_csv(output_file, index=False)
    print(f"已创建标签文件: {output_file}")
    print(f"标签分布: {df['label'].value_counts().to_dict()}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='创建标签文件')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-fs/prov-gigapath/data',
                        help='包含ndpi文件的数据目录')
    parser.add_argument('--output_file', type=str, default='/root/gigapath_output/labels.csv',
                        help='输出标签文件的路径')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数量')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    create_labels_file(args.data_dir, args.output_file, args.num_classes)

if __name__ == '__main__':
    main()