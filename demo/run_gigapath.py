'''
加载 和 使用模型。可以看成完整的教程
'''
import os
import huggingface_hub
from gigapath.pipeline import tile_one_slide
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import run_inference_with_tile_encoder
from gigapath.pipeline import run_inference_with_slide_encoder
 
 
os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"
 
 
# 下载演示图像，并进行切割
local_dir = os.path.join(os.path.expanduser("~"), "cache/")
#huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=False)
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