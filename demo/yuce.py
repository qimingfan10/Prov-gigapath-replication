import os
import huggingface_hub
from gigapath.pipeline import tile_one_slide
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import run_inference_with_tile_encoder
from gigapath.pipeline import run_inference_with_slide_encoder
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Rectangle
import time

start_time = time.time()  # 获取当前时间戳 (秒)
os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"


# 下载演示图像，并进行切割
local_dir = os.path.join(os.path.expanduser("~"), "cache/")
huggingface_hub.hf_hub_download(repo_id="prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=False)
slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")# 如果想预测自己的图像可以更改下面的值
slide_path = slide_path
# 注：Prov-GigaPath 使用以 0.5 MPP 预处理的载玻片进行训练。确保为 0.5 MPP 使用适当的级别。
# 这里使用  1_slide_mpp_check.py  的方法检查图像
# 检查图像 mpp = 0.5 的层数
tmp_dir = './outputs/preprocessing/' # 分割后的图像保存的路径
target_mpp = 0.5
level = find_level_for_target_mpp(slide_path, target_mpp)
if level is not None:
    print(f"mpp为0.5的层数是: {level}")
else:
    raise EOFError('没有找到mpp为0.5的层数')

tile_one_slide(slide_path, save_dir=tmp_dir, level=level)


# 加载图片
slide_dir = "outputs/preprocessing/output/" + os.path.basename(slide_path) + "/"
image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]

print(f"找到 {len(image_paths)} 个图像")

# 模型加载
tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

# 运行平铺级别推理
tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
for k in tile_encoder_outputs.keys():
    print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")


# run inference with the slide encoder
slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
print(slide_embeds.keys())

for i in slide_embeds:
    print(i)
    print(slide_embeds[i].shape)

# 基因预测模型加载和预测
class GeneMutationPredictionModel(torch.nn.Module):
    def __init__(self, slide_encoder):
        super().__init__()
        self.slide_encoder = slide_encoder
        self.prediction_head = torch.nn.Linear(768, 19)  # 19个biomarkers (包括TMB_High)

    def forward(self, tile_embeds):
        # 修复参数传递方式，确保使用正确的参数名
        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=self.slide_encoder,
                                                        tile_embeds=tile_embeds['tile_embeds'],
                                                        coords=tile_embeds['coords'])
        # 获取最后一层的嵌入向量用于预测
        last_layer_embed = slide_embeds['last_layer_embed']
        logits = self.prediction_head(last_layer_embed)
        return logits

prediction_model = GeneMutationPredictionModel(slide_encoder_model)
prediction_model.eval() # 设置为评估模式

# 进行预测
with torch.no_grad():
    logits = prediction_model(tile_encoder_outputs)
    predictions = torch.sigmoid(logits) # 使用 sigmoid 函数进行多标签分类，对于二分类任务可以使用 softmax


# 后处理和结果展示
labels = ['EGFR', 'KRAS', 'BRAF', 'PIK3CA', 'TP53', 'PTEN', 'ALK_Fusion', 'ROS1_Fusion', 
          'NTRK_Fusion', 'ERBB2', 'MET', 'RET_Fusion', 'CTNNB1', 'APC', 'VHL', 
          'BRCA1', 'BRCA2', 'MSI_High', 'TMB_High']
threshold = 0.5

predicted_labels = []
for i in range(predictions.shape[1]):
    if labels[i] == 'TMB_High':
        if predictions[0, i] >= threshold:
            predicted_labels.append(('TMB_High', float(predictions[0, i])))
        else:
            predicted_labels.append(('TMB_Low', float(1 - predictions[0, i]))) # 输出 TMB_Low 的概率
    elif labels[i] == 'MSI_High':
        if predictions[0, i] >= threshold:
            predicted_labels.append(('MSI_High', float(predictions[0, i])))
        else:
            predicted_labels.append(('MSI_Stable', float(1 - predictions[0, i]))) # 输出 MSI_Stable 的概率
    elif predictions[0, i] >= threshold:
        if labels[i].endswith('_Fusion'):
            predicted_labels.append((labels[i], float(predictions[0, i])))
        else:
            predicted_labels.append((f"{labels[i]}_Mutated", float(predictions[0, i])))

print("\n基因突变预测结果:")
for label, probability in predicted_labels:
    print(f"{label}: {probability:.4f}")

# 可视化展示和保存
output_image_path = "prediction_result.png"

# 修改：不直接加载原始大图，而是使用已经切割好的小图作为示例
try:
    # 使用第一个切割好的图像作为示例
    sample_tile_image = Image.open(image_paths[0]) if image_paths else None
except Exception as e:
    print(f"无法加载示例图像: {e}")
    sample_tile_image = None

fig, axes = plt.subplots(1, 2, figsize=(15, 5)) # 创建包含两个子图的Figure

# 子图 1: 展示示例切片图像
if sample_tile_image is not None:
    axes[0].imshow(sample_tile_image)
    axes[0].set_title('示例切片图像')
else:
    axes[0].text(0.5, 0.5, '无法加载示例图像', ha='center', va='center')
    axes[0].set_title('示例图像加载失败')
axes[0].axis('off') # 关闭坐标轴

# 子图 2: 绘制基因突变预测结果表格
cell_text = [[label, f"{probability:.4f}"] for label, probability in predicted_labels]
col_labels = ["基因/生物标志物", "预测概率"]
table = axes[1].table(cellText=cell_text, colLabels=col_labels, cellLoc = 'center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # 适当调整表格大小
axes[1].set_title('基因突变预测结果', fontweight='bold', fontsize=12)
axes[1].axis('off') # 关闭坐标轴

plt.tight_layout() # 调整子图布局，避免重叠
plt.savefig(output_image_path, dpi=300, bbox_inches='tight') # 保存图片，bbox_inches='tight' 避免裁剪
plt.show() # 显示图像

print(f"\n可视化结果已保存到: {output_image_path}")
end_time = time.time()   # 获取代码结束时的时间戳
elapsed_time = end_time - start_time  # 计算时间差 (秒)

print(f"代码运行时间: {elapsed_time:.4f} 秒") # 格式化输出，保留 4 位小数