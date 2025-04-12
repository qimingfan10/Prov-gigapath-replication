import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import openslide
from openslide import OpenSlide
import sys

class NDPIViewer:
    def __init__(self, ndpi_path):
        """
        初始化 NDPI 查看器
        
        Args:
            ndpi_path (str): NDPI 文件的路径
        """
        self.slide = OpenSlide(ndpi_path)
        self.filename = os.path.basename(ndpi_path)
        
        # 获取图像尺寸和层级信息
        self.dimensions = self.slide.dimensions
        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        
        # 初始化查看参数
        self.current_level = self.level_count - 1  # 从最低分辨率开始
        self.x_pos = 0
        self.y_pos = 0
        self.zoom_factor = 1.0
        
        # 打印图像信息
        print(f"文件名: {self.filename}")
        print(f"图像尺寸: {self.dimensions[0]} x {self.dimensions[1]}")
        print(f"层级数量: {self.level_count}")
        for i in range(self.level_count):
            print(f"  层级 {i}: {self.level_dimensions[i][0]} x {self.level_dimensions[i][1]} "
                  f"(下采样率: {self.level_downsamples[i]})")
        
        # 创建图形界面
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # 添加滑块控制
        ax_level = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_x = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_y = plt.axes([0.25, 0.05, 0.65, 0.03])
        
        # 创建滑块
        self.level_slider = Slider(
            ax_level, '分辨率层级', 0, self.level_count-1, 
            valinit=self.current_level, valstep=1
        )
        
        max_x = self.level_dimensions[self.current_level][0] - 1000
        max_y = self.level_dimensions[self.current_level][1] - 1000
        
        self.x_slider = Slider(
            ax_x, 'X 位置', 0, max_x, valinit=self.x_pos, valstep=100
        )
        
        self.y_slider = Slider(
            ax_y, 'Y 位置', 0, max_y, valinit=self.y_pos, valstep=100
        )
        
        # 添加滑块事件处理
        self.level_slider.on_changed(self.update_level)
        self.x_slider.on_changed(self.update_position)
        self.y_slider.on_changed(self.update_position)
        
        # 添加按钮
        ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
        self.reset_button = Button(ax_reset, '重置')
        self.reset_button.on_clicked(self.reset)
        
        # 显示初始图像
        self.update_image()
        
        # 添加鼠标事件
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
    def update_level(self, val):
        """更新分辨率层级"""
        self.current_level = int(val)
        
        # 更新 X 和 Y 滑块的范围
        max_x = max(0, self.level_dimensions[self.current_level][0] - 1000)
        max_y = max(0, self.level_dimensions[self.current_level][1] - 1000)
        
        # 调整当前位置以适应新的层级
        scale_factor = self.level_downsamples[self.current_level] / self.level_downsamples[int(self.level_slider.val)]
        self.x_pos = min(int(self.x_pos * scale_factor), max_x)
        self.y_pos = min(int(self.y_pos * scale_factor), max_y)
        
        self.x_slider.valmax = max_x
        self.y_slider.valmax = max_y
        self.x_slider.ax.set_xlim(0, max_x)
        self.y_slider.ax.set_xlim(0, max_y)
        self.x_slider.set_val(self.x_pos)
        self.y_slider.set_val(self.y_pos)
        
        self.update_image()
        
    def update_position(self, val):
        """更新查看位置"""
        self.x_pos = int(self.x_slider.val)
        self.y_pos = int(self.y_slider.val)
        self.update_image()
        
    def update_image(self):
        """更新显示的图像"""
        # 计算要读取的区域大小
        view_size = 1000
        level_size = self.level_dimensions[self.current_level]
        
        # 确保不超出图像边界
        x = min(self.x_pos, level_size[0] - view_size)
        y = min(self.y_pos, level_size[1] - view_size)
        x = max(0, x)
        y = max(0, y)
        
        # 读取图像区域
        region = self.slide.read_region(
            (int(x * self.level_downsamples[self.current_level]), 
             int(y * self.level_downsamples[self.current_level])),
            self.current_level,
            (min(view_size, level_size[0] - x), 
             min(view_size, level_size[1] - y))
        )
        
        # 转换为RGB数组
        region_array = np.array(region)[:, :, :3]
        
        # 更新图像
        self.ax.clear()
        self.ax.imshow(region_array)
        self.ax.set_title(f"{self.filename} - 层级 {self.current_level} "
                         f"({self.level_dimensions[self.current_level][0]}x"
                         f"{self.level_dimensions[self.current_level][1]})")
        self.ax.set_xlabel(f"X: {x} - {x + region_array.shape[1]}")
        self.ax.set_ylabel(f"Y: {y} - {y + region_array.shape[0]}")
        self.fig.canvas.draw_idle()
        
    def reset(self, event):
        """重置查看参数"""
        self.current_level = self.level_count - 1
        self.x_pos = 0
        self.y_pos = 0
        self.zoom_factor = 1.0
        
        self.level_slider.set_val(self.current_level)
        self.x_slider.set_val(self.x_pos)
        self.y_slider.set_val(self.y_pos)
        
        self.update_image()
        
    def on_scroll(self, event):
        """鼠标滚轮事件处理"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 'up':
            # 放大 - 降低层级
            if self.current_level > 0:
                self.current_level -= 1
                self.level_slider.set_val(self.current_level)
        elif event.button == 'down':
            # 缩小 - 提高层级
            if self.current_level < self.level_count - 1:
                self.current_level += 1
                self.level_slider.set_val(self.current_level)
    
    def on_press(self, event):
        """鼠标按下事件处理"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # 左键
            self.is_dragging = True
            self.drag_start_x = event.xdata
            self.drag_start_y = event.ydata
    
    def on_release(self, event):
        """鼠标释放事件处理"""
        self.is_dragging = False
    
    def on_motion(self, event):
        """鼠标移动事件处理"""
        if not self.is_dragging or event.inaxes != self.ax:
            return
            
        # 计算拖动距离
        dx = self.drag_start_x - event.xdata
        dy = self.drag_start_y - event.ydata
        
        if dx == 0 and dy == 0:
            return
            
        # 更新位置
        self.x_pos = min(max(0, self.x_pos + int(dx)), self.x_slider.valmax)
        self.y_pos = min(max(0, self.y_pos + int(dy)), self.y_slider.valmax)
        
        # 更新滑块
        self.x_slider.set_val(self.x_pos)
        self.y_slider.set_val(self.y_pos)
        
        # 重置拖动起点
        self.drag_start_x = event.xdata
        self.drag_start_y = event.ydata

def main():
    if len(sys.argv) < 2:
        print("用法: python ndpi_viewer.py <ndpi文件路径>")
        return
        
    ndpi_path = sys.argv[1]
    if not os.path.exists(ndpi_path):
        print(f"错误: 文件 '{ndpi_path}' 不存在")
        return
        
    if not ndpi_path.lower().endswith('.ndpi'):
        print(f"警告: 文件 '{ndpi_path}' 可能不是NDPI格式")
        
    try:
        viewer = NDPIViewer(ndpi_path)
        plt.show()
    except Exception as e:
        print(f"错误: {e}")
        
if __name__ == "__main__":
    main() 