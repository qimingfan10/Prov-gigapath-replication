import os
import timm
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import List
import cv2
import matplotlib.pyplot as plt
 
# 令牌设置
os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"
 
# 常量配置
DEVICE = 'cuda'
NUMBER_COMPONENTS = 2 
image_paths = [
    "./images/01581x_25327y.png",
    "./images/01581x_25583y.png",
    #'images/P301503_53248_50176_11264_23552_11776_24064.jpg'
]
output_path = './outputs'
 
 
# 模型创建
model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
 
device = DEVICE
model.to(device)
 
# 用于对图像进行归一化
# 统一图像大小
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
 
def load_and_preprocess_image(image_path: str) -> Image.Image:
    '''
    用于打开图像
    '''
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img
 
pca = PCA(n_components=3) # 设置主成分降维的数量
scaler = MinMaxScaler(clip=True)
 
def process_images(images: List[Image.Image], background_threshold: float = 0.5, larger_pca_as_fg: bool = False) -> List[np.ndarray]:
    imgs_tensor = torch.stack([transform(img).to(device) for img in images])
    with torch.no_grad():
        # 获取模型中间特征，intermediates_only=True 表示只获取中间特征
        intermediate_features = model.forward_intermediates(imgs_tensor, intermediates_only=True)
        # 将最后一层的特征转换为合适的形状
        features = intermediate_features[-1].permute(0, 2, 3, 1).reshape(-1, 1536).cpu()
    # 对特征进行 PCA 降维，并使用缩放器进行缩放
    pca_features = scaler.fit_transform(pca.fit_transform(features))
    # 根据 larger_pca_as_fg 标志判断前景索引
    if larger_pca_as_fg:
        # 如果 larger_pca_as_fg 为 True，则前景是大于背景阈值的
        fg_indices = pca_features[:, 0] > background_threshold
    else:
        # 否则，前景是小于背景阈值的
        fg_indices = pca_features[:, 0] < background_threshold
    # 使用前景索引提取前景特征
    fg_features = pca.fit_transform(features[fg_indices])
    # 对前景特征进行缩放
    scaler.fit(fg_features)
    normalized_features = scaler.transform(fg_features)
    # 准备结果图像，假设每张图像被划分为 14x14 的小块
    result_img = np.zeros((imgs_tensor.size(0) * 196, 3))
    # 将归一化的前景特征填入结果图像
    result_img[fg_indices] = normalized_features
    # 将 imgs_tensor 移动到 CPU 以便后续处理
    imgs_tensor = imgs_tensor.cpu()
    transformed_imgs = []
    # 对每张图像进行处理，将张量转换为 NumPy 数组并反归一化
    for i, img in enumerate(imgs_tensor):
        img_np = img.permute(1, 2, 0).numpy()  # 转换维度为 (高, 宽, 通道)
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # 反归一化
        img_np = (img_np * 255).astype(np.uint8)  # 转换为 8 位无符号整数
        transformed_imgs.append(img_np)  # 添加到结果列表
    # 将结果图像重塑为每张输入图像的形状
    results = [result_img.reshape(imgs_tensor.size(0), 14, 14, 3)[i] for i in range(len(images))]
    # 返回结果，包含处理后的图像、变换后的图像和 PCA 特征
    return results, transformed_imgs, pca_features
 
images = [load_and_preprocess_image(path) for path in image_paths]
results, transformed_imgs, pca_features = process_images(images, larger_pca_as_fg=False)
 
 
# 查看主成分分析的结果
def create_overlay_image(original, result, alpha=0.3):
    # Resize result to match the original image size
    result_resized = cv2.resize(result, (original.shape[1], original.shape[0]))
    overlay = (alpha * original + (1 - alpha) * result_resized * 255).astype(np.uint8)
    return overlay
 
num_images = len(transformed_imgs)
 
fig, axes = plt.subplots(num_images, 3, figsize=(9, 3 * num_images))
 
for i, (image, result) in enumerate(zip(transformed_imgs, results)):
    overlay = create_overlay_image(image, result)
 
    # Original image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Original Image {i+1}")
    axes[i, 0].axis('off')
 
    # PCA result image
    axes[i, 1].imshow(result)
    axes[i, 1].set_title(f"Foreground-Only PCA for Image {i+1}")
    axes[i, 1].axis('off')
 
    # Overlay image
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f"Overlay for Image {i+1}")
    axes[i, 2].axis('off')
 
#fig.suptitle('PCA Visualizations', fontsize=20)
 
plt.tight_layout()
plt.savefig(os.path.join(output_path,'PCA.png'))
plt.close()
 
plt.figure(figsize=(9, 3))
 
plt.subplot(1, 3, 1)
plt.hist(pca_features[:, 0], bins=30, alpha=0.7)
plt.title('Histogram of 1st PCA Component')
 
plt.subplot(1, 3, 2)
plt.hist(pca_features[:, 1], bins=30, alpha=0.7)
plt.title('Histogram of 2nd PCA Component')
 
plt.subplot(1, 3, 3)
plt.hist(pca_features[:, 2], bins=30, alpha=0.7)
plt.title('Histogram of 3rd PCA Component')
 
plt.tight_layout()
plt.savefig(os.path.join(output_path,'histogram_of_PCA.png'))
plt.close()
 
patch_h, patch_w = 14, 14
 
fig, axes = plt.subplots(num_images, 4, figsize=(12, 3 * num_images))
 
for i in range(num_images):
    # Original image
    axes[i, 0].imshow(transformed_imgs[i])
    axes[i, 0].set_title(f"Original Image {i+1}")
    axes[i, 0].axis('off')
 
    # First component
    component1 = pca_features[i * patch_h * patch_w : (i + 1) * patch_h * patch_w, 0].reshape(patch_h, patch_w)
    axes[i, 1].imshow(component1, cmap='viridis_r')
    axes[i, 1].set_title(f'1st Component')
    axes[i, 1].axis('off')
 
    # Second component
    component2 = pca_features[i * patch_h * patch_w : (i + 1) * patch_h * patch_w, 1].reshape(patch_h, patch_w)
    axes[i, 2].imshow(component2, cmap='viridis_r')
    axes[i, 2].set_title(f'2nd Component')
    axes[i, 2].axis('off')
 
    # Third component
    component3 = pca_features[i * patch_h * patch_w : (i + 1) * patch_h * patch_w, 2].reshape(patch_h, patch_w)
    axes[i, 3].imshow(component3, cmap='viridis_r')
    axes[i, 3].set_title(f'3rd Component')
    axes[i, 3].axis('off')
 
    # RGB visualization
    # rgb_image = np.stack((component3, component2, component1), axis=-1)
    # axes[i, 4].imshow(rgb_image)
    # axes[i, 4].set_title(f'RGB Visualization')
    # axes[i, 4].axis('off')
 
fig.suptitle('PCA Visualization', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_path,'PCA_Visualization.png'))
plt.close()