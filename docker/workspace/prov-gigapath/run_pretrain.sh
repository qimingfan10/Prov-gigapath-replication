#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:.  # 将当前目录添加到 PYTHONPATH

# 创建输出目录
OUTPUT_DIR="./output"  # 在当前目录下创建 output 文件夹
mkdir -p $OUTPUT_DIR
# 运行预训练流程
echo "开始GigaPath模型预训练..."
python pretrain_gigapath.py \
    --data_dir ./data \
    --output_dir "$OUTPUT_DIR" \
    --level 0 \
    --tile_size 256 \
    --batch_size 64 \
    --num_epochs_tile 100 \
    --num_epochs_slide 50 \
    --learning_rate 1e-4 \
    --mask_ratio 0.75

echo "预训练完成！模型保存在 $OUTPUT_DIR"
