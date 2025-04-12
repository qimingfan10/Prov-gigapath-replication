import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from openslide import OpenSlide
import argparse
from tqdm import tqdm

def extract_ndpi_regions(ndpi_path, output_dir=None, regions_per_level=5, region_size=1024):
    """
    从NDPI文件中提取并保存不同层级的图像区域
    
    Args:
        ndpi_path (str): NDPI文件路径
        output_dir (str): 输出目录，默认为当前目录下的文件名_regions
        regions_per_level (int): 每个层级提取的区域数量
        region_size (int): 提取区域的大小（像素）
    """
    # 打开NDPI文件
    try:
        slide = OpenSlide(ndpi_path)
    except Exception as e:
        print(f"错误: 无法打开NDPI文件 '{ndpi_path}': {e}")
        return
    
    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(ndpi_path))[0]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = f"{filename}_regions"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像信息
    dimensions = slide.dimensions
    level_count = slide.level_count
    level_dimensions = slide.level_dimensions
    level_downsamples = slide.level_downsamples
    
    # 打印图像信息
    print(f"文件名: {filename}")
    print(f"图像尺寸: {dimensions[0]} x {dimensions[1]}")
    print(f"层级数量: {level_count}")
    for i in range(level_count):
        print(f"  层级 {i}: {level_dimensions[i][0]} x {level_dimensions[i][1]} "
              f"(下采样率: {level_downsamples[i]})")
    
    # 保存缩略图
    try:
        thumbnail_size = 1024
        thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
        thumbnail_path = os.path.join(output_dir, f"{filename}_thumbnail.png")
        thumbnail.save(thumbnail_path)
        print(f"已保存缩略图: {thumbnail_path}")
    except Exception as e:
        print(f"警告: 无法保存缩略图: {e}")
    
    # 为每个层级提取区域
    for level in range(level_count):
        level_dir = os.path.join(output_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)
        
        width, height = level_dimensions[level]
        
        # 调整区域大小，确保不超过该层级的尺寸
        actual_region_size = min(region_size, width, height)
        
        print(f"正在处理层级 {level} ({width}x{height})...")
        
        # 生成随机位置
        np.random.seed(42)  # 使结果可重现
        
        max_x = max(0, width - actual_region_size)
        max_y = max(0, height - actual_region_size)
        
        # 确保至少有一个区域
        if regions_per_level <= 0:
            regions_per_level = 1
        
        # 生成均匀分布的位置
        if regions_per_level == 1:
            # 如果只提取一个区域，取中心
            x_positions = [max_x // 2]
            y_positions = [max_y // 2]
        else:
            # 否则均匀分布
            x_positions = np.linspace(0, max_x, regions_per_level).astype(int)
            y_positions = np.linspace(0, max_y, regions_per_level).astype(int)
        
        # 提取并保存区域
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                region_path = os.path.join(level_dir, f"region_x{x}_y{y}.png")
                
                # 如果文件已存在，跳过
                if os.path.exists(region_path):
                    continue
                
                try:
                    # 读取区域
                    region = slide.read_region(
                        (int(x * level_downsamples[level]), 
                         int(y * level_downsamples[level])),
                        level,
                        (min(actual_region_size, width - x), 
                         min(actual_region_size, height - y))
                    )
                    
                    # 转换为RGB并保存
                    region_rgb = region.convert('RGB')
                    region_rgb.save(region_path)
                except Exception as e:
                    print(f"警告: 无法保存区域 (level={level}, x={x}, y={y}): {e}")
        
        print(f"已保存 {len(x_positions) * len(y_positions)} 个区域到 {level_dir}")
    
    # 关闭文件
    slide.close()
    print(f"处理完成。所有图像已保存到 {output_dir}")

def extract_ndpi_full_levels(ndpi_path, output_dir=None, max_size=8192):
    """
    从NDPI文件中提取并保存每个层级的完整图像
    
    Args:
        ndpi_path (str): NDPI文件路径
        output_dir (str): 输出目录，默认为当前目录下的文件名_levels
        max_size (int): 保存完整层级的最大尺寸限制
    """
    # 打开NDPI文件
    try:
        slide = OpenSlide(ndpi_path)
    except Exception as e:
        print(f"错误: 无法打开NDPI文件 '{ndpi_path}': {e}")
        return
    
    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(ndpi_path))[0]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = f"{filename}_levels"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像信息
    level_count = slide.level_count
    level_dimensions = slide.level_dimensions
    
    # 打印图像信息
    print(f"文件名: {filename}")
    print(f"层级数量: {level_count}")
    
    # 保存每个层级的完整图像
    for level in range(level_count):
        width, height = level_dimensions[level]
        print(f"处理层级 {level} ({width}x{height})...")
        
        # 检查尺寸是否超过限制
        if width > max_size or height > max_size:
            print(f"  跳过: 尺寸超过限制 ({max_size}x{max_size})")
            continue
        
        # 设置输出路径
        level_path = os.path.join(output_dir, f"{filename}_level_{level}.png")
        
        # 如果文件已存在，跳过
        if os.path.exists(level_path):
            print(f"  跳过: 文件已存在 {level_path}")
            continue
        
        try:
            # 读取整个层级
            image = slide.read_region((0, 0), level, (width, height))
            
            # 转换为RGB并保存
            image_rgb = image.convert('RGB')
            image_rgb.save(level_path)
            print(f"  已保存: {level_path}")
        except Exception as e:
            print(f"  错误: 无法保存层级 {level}: {e}")
    
    # 关闭文件
    slide.close()
    print(f"处理完成。所有层级图像已保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='NDPI病理图像提取工具')
    parser.add_argument('ndpi_path', help='NDPI文件路径')
    parser.add_argument('--output', '-o', help='输出目录')
    parser.add_argument('--mode', '-m', choices=['regions', 'levels', 'both'], default='both',
                        help='提取模式: regions=提取区域, levels=提取完整层级, both=两者都提取')
    parser.add_argument('--regions', '-r', type=int, default=3,
                        help='每个层级提取的区域数量 (默认: 3)')
    parser.add_argument('--size', '-s', type=int, default=1024,
                        help='提取区域的大小 (默认: 1024)')
    parser.add_argument('--max-size', type=int, default=8192,
                        help='保存完整层级的最大尺寸限制 (默认: 8192)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ndpi_path):
        print(f"错误: 文件 '{args.ndpi_path}' 不存在")
        return
    
    if args.mode in ['regions', 'both']:
        output_dir = args.output if args.output else None
        extract_ndpi_regions(args.ndpi_path, output_dir, args.regions, args.size)
    
    if args.mode in ['levels', 'both']:
        output_dir = args.output if args.output else None
        extract_ndpi_full_levels(args.ndpi_path, output_dir, args.max_size)

if __name__ == "__main__":
    main() 