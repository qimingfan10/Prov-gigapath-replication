import os
import huggingface_hub
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.environ["HF_TOKEN"] = "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"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

# --------  数据加载部分 (修改以平均 tile embeddings) --------
embedding_dir = "GigaPath_PANDA_embeddings/h5_files"
embedding_paths = [os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) if f.endswith('.h5')]

slide_embeddings = [] #  修改：存储 slide-level embeddings
filenames = []
count = 0
for path in embedding_paths:
    if count >= 10:
        break
    with h5py.File(path, 'r') as hf:
        if 'features' in hf:
            tile_embeddings = np.array(hf['features'][:]) # 加载 tile-level embeddings
            print(f"加载文件: {os.path.basename(path)}, tile embedding shape: {tile_embeddings.shape}") # 打印 tile embedding shape
            slide_embedding = np.mean(tile_embeddings, axis=0) #  关键修改：平均 tile embeddings 得到 slide-level embedding
            print(f"平均后 slide embedding shape: {slide_embedding.shape}") # 打印 slide embedding shape
            slide_embeddings.append(slide_embedding) #  将 slide-level embedding 加入列表
            filenames.append(os.path.basename(path))
            count += 1
        else:
            print(f"警告: 文件 {path} 中没有找到 'features' 键，请检查 h5 文件结构。")

slide_embeddings = np.array(slide_embeddings) #  转换为 numpy array (之前已经转换过了，但为了代码逻辑更清晰，再转换一次)

print(f"加载了 {len(slide_embeddings)} 个 slide 的 embedding，形状为: {slide_embeddings.shape}") # 打印最终 slide_embeddings 的形状

# --------  标签和数据集划分部分 (保持不变，但请替换为真实标签) --------
num_samples = slide_embeddings.shape[0]
embedding_dim = slide_embeddings.shape[1] # embedding_dim 现在应该是 1536
dummy_labels = np.random.randint(0, 6, num_samples)

X_train, X_test, y_train, y_test = train_test_split(slide_embeddings, dummy_labels, test_size=0.2, random_state=42)

# --------  模型训练和评估部分 (保持不变) --------
classifier = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"示例六分类器在随机测试集上的准确率: {accuracy:.2f}")

# --------  预测每个输入文件的癌症类别 --------
label_names = ['癌症类别 0', '癌症类别 1', '癌症类别 2', '癌症类别 3', '癌症类别 4', '癌症类别 5']

print("\n-----  预测每个输入文件的癌症类别  -----")
for i in range(len(slide_embeddings)): # 循环遍历 slide-level embeddings
    input_slide_embedding = slide_embeddings[i].reshape(1, -1) #  reshape 成模型需要的输入形状 (1, embedding_dim)
    print(f"预测前 slide_embeddings[{i}] shape: {slide_embeddings[i].shape}, input_slide_embedding shape: {input_slide_embedding.shape}") # 打印预测前的形状
    predicted_label_index = classifier.predict(input_slide_embedding)[0]
    predicted_label = label_names[predicted_label_index]
    filename = filenames[i]
    print(f"文件: {filename}, 预测癌症分类标签: {predicted_label}")