import os
import openslide
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def show_whole_slide(slide_path, output_path=None, thumbnail_size=1024):
    """
    展示全切片病理图像的缩略图
    
    参数:
        slide_path: 切片文件路径
        output_path: 输出图像保存路径，如果为None则只显示不保存
        thumbnail_size: 缩略图的最大尺寸（宽或高的最大值）
    """
    try:
        # 使用OpenSlide打开切片
        slide = openslide.OpenSlide(slide_path)
        
        # 获取切片的基本信息
        dimensions = slide.dimensions
        level_count = slide.level_count
        level_dimensions = slide.level_dimensions
        level_downsamples = slide.level_downsamples
        
        print(f"切片尺寸: {dimensions[0]} x {dimensions[1]} 像素")
        print(f"层级数量: {level_count}")
        for i in range(level_count):
            print(f"  层级 {i}: {level_dimensions[i][0]} x {level_dimensions[i][1]} 像素 (下采样率: {level_downsamples[i]})")
        
        # 获取切片的属性
        properties = slide.properties
        print("\n切片属性:")
        for key in properties:
            print(f"  {key}: {properties[key]}")
        
        # 获取合适的层级用于生成缩略图
        # 选择尺寸略大于thumbnail_size的层级
        suitable_level = 0
        for i in range(level_count):
            if max(level_dimensions[i]) < thumbnail_size * 2:
                suitable_level = i
                break
        
        # 从选定的层级读取图像
        level_image = slide.read_region((0, 0), suitable_level, level_dimensions[suitable_level])
        level_image = level_image.convert('RGB')
        
        # 调整图像大小以适应thumbnail_size
        width, height = level_image.size
        if width > height:
            new_width = thumbnail_size
            new_height = int(height * (thumbnail_size / width))
        else:
            new_height = thumbnail_size
            new_width = int(width * (thumbnail_size / height))
        
        thumbnail = level_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建多层级预览图
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # 显示缩略图
        axes[0].imshow(np.array(thumbnail))
        axes[0].set_title(f'全切片缩略图 (层级 {suitable_level})')
        axes[0].axis('off')
        
        # 显示多层级预览
        # 选择4个不同层级的小区域进行展示
        region_size = 256
        x_center, y_center = dimensions[0] // 2, dimensions[1] // 2
        
        # 选择一个合适的层级，使得区域大小合适
        region_level = min(2, level_count - 1)  # 使用较高分辨率的层级
        region_scale = int(level_downsamples[region_level])
        
        # 读取区域图像
        region_x = max(0, x_center - (region_size * region_scale) // 2)
        region_y = max(0, y_center - (region_size * region_scale) // 2)
        region_image = slide.read_region(
            (region_x, region_y), 
            region_level, 
            (region_size, region_size)
        ).convert('RGB')
        
        # 显示区域图像
        axes[1].imshow(np.array(region_image))
        axes[1].set_title(f'中心区域 (层级 {region_level})')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {output_path}")
        
        plt.show()
        
        # 关闭切片
        slide.close()
        
    except Exception as e:
        print(f"处理切片时出错: {e}")

if __name__ == "__main__":
    # 设置切片路径
    slide_path = "/root/cache/sample_data/PROV-000-000001.ndpi"
    output_path = "whole_slide_preview.png"
    
    # 确保文件存在
    if not os.path.exists(slide_path):
        print(f"错误: 文件 {slide_path} 不存在")
    else:
        show_whole_slide(slide_path, output_path) 