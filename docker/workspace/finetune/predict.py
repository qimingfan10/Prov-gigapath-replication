import os
import torch
import pandas as pd
import numpy as np

from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_loader
from datasets.slide_datatset import SlideDataset
from gigapath.classification_head import get_model
from gigapath.slide_encoder import create_model
import time

start_time = time.time()  # 获取当前时间戳 (秒)
def predict(checkpoint_path, dataset_csv, root_path, task_cfg_path, save_dir, exp_name):
    """
    Predicts on a dataset using a trained model checkpoint.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint (.pth file).
        dataset_csv (str): Path to the dataset CSV file.
        root_path (str): Root path to the image feature files.
        task_cfg_path (str): Path to the task configuration YAML file.
        save_dir (str): Directory to save the prediction results.
        exp_name (str): Experiment name for saving results.
    """
    
    args = get_finetune_params()
    # Override arguments with function inputs
    args.checkpoint_path = checkpoint_path
    args.dataset_csv = dataset_csv
    args.root_path = root_path
    args.task_cfg_path = task_cfg_path
    args.save_dir = save_dir
    args.exp_name = exp_name
    # **不再显式设置 pretrained，让 get_model 创建默认模型**
    # args.pretrained = 'hf_hub:prov-gigapath/prov-gigapath'
    print("Prediction arguments:")
    print(args)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed (for reproducibility, though not strictly necessary for prediction)
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')

    # **显式设置 model_arch**
    args.model_arch = args.task_config.get('model_arch', args.model_arch)  # 从 task_config 读取 model_arch，如果 task_config 中没有，则使用 args 中已有的 model_arch
    print(f"Model architecture being used: {args.model_arch}")  # 打印 model_arch

    # set the experiment save directory for predictions
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name, 'predictions')
    os.makedirs(args.save_dir, exist_ok=True)
    print('Setting save directory for predictions: {}'.format(args.save_dir))

    # set up the dataset
    try:
        dataset = pd.read_csv(args.dataset_csv)
    except FileNotFoundError:
        print(f"Error: Dataset CSV file {args.dataset_csv} not found.")
        return
    DatasetClass = SlideDataset

    # Instantiate dataset for prediction (no splits needed)
    predict_data = DatasetClass(dataset, args.root_path, dataset['slide_id'].tolist(), args.task_config, split_key='slide_id')  # Use all slide_ids for prediction
    args.n_classes = predict_data.n_classes
    print(f"Number of classes: {args.n_classes}")

    # get the dataloader for prediction
    predict_loader = get_loader(predict_data, None, None, task_config=args.task_config, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)[0]  # Only need train_loader (which is actually predict_loader here)

    

    # **MODIFICATION: 修改检查点中的键名**
    print('Loading checkpoint from: {}'.format(args.checkpoint_path))
    try:
        checkpoint_state_dict = torch.load(args.checkpoint_path, map_location=args.device)
        print("Checkpoint keys:", list(checkpoint_state_dict.keys())[:5], "...")
    except FileNotFoundError:
        print(f"Error: Checkpoint file {args.checkpoint_path} not found.")
        return

    
    # 创建一个新的状态字典，用于加载到模型
    new_state_dict = {}
    for k, v in checkpoint_state_dict.items():
        if k.startswith('slide_encoder.'):
            # 将 slide_encoder.xxx 转换为 xxx
            new_key = k[len('slide_encoder.'):]
            new_state_dict[new_key] = v
        elif k.startswith('classifier.'):
            # 保持 classifier.xxx 不变
            new_state_dict[k] = v
    # 设置模型
    print(f"Model architecture being used BEFORE get_model: {args.model_arch}")
    model = get_model(**vars(args))
    model = model.to(args.device)

    # 使用 strict=False 加载模型，允许缺失和多余的键
    model.slide_encoder.load_state_dict(new_state_dict, strict=False)
    print("Model slide_encoder loaded with strict=False")

    # 加载分类器部分
    classifier_dict = {k: v for k, v in checkpoint_state_dict.items() if k.startswith('classifier.')}
    if classifier_dict:
        model.classifier.load_state_dict(classifier_dict, strict=False)
        print("Model classifier loaded with strict=False")

    model.eval()  # Set model to evaluation mode

    results = []

    # 创建 fp16 scaler
    fp16_scaler = torch.cuda.amp.GradScaler() if args.device != 'cpu' else None

    with torch.no_grad():  # Disable gradient calculations during inference
        for batch_idx, batch in enumerate(predict_loader):
            # 只处理前 500 个批次
            if batch_idx >= 1:
                print(f"已处理 500 个批次，根据要求停止处理")
                break
            
            try:
                images, img_coords, labels, slide_ids = batch['imgs'], batch['coords'], batch['labels'], batch['slide_id']
                images = images.to(args.device, non_blocking=True)
                img_coords = img_coords.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True).long()

                # 使用 autocast 启用混合精度
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    # Forward pass
                    logits = model(images, img_coords)
                
                if args.task_config.get('setting', 'multi_class') == 'multi_label':
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:  # multi_class or binary
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                labels_cpu = labels.cpu().numpy()
                slide_ids_cpu = slide_ids

                for i in range(len(slide_ids_cpu)):
                    result_dict = {
                        'slide_id': slide_ids_cpu[i],
                        'label': labels_cpu[i].tolist() if labels_cpu.ndim > 1 else labels_cpu[i],  # Handle both multi-label and multi_class/binary labels
                        'probabilities': probs[i].tolist()
                    }
                    results.append(result_dict)
                print(f"Batch {batch_idx + 1}/{len(predict_loader)} processed.")
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    output_csv_path = os.path.join(args.save_dir, 'predictions.csv')
    results_df.to_csv(output_csv_path, index=False)

    print('Predictions saved in: {}'.format(output_csv_path))
    print('Done with prediction!')
    

if __name__ == '__main__':
    # Example usage: Replace with your actual paths
    checkpoint_path = 'finetune/checkpoint.pt' # Path to your .pth file
    dataset_csv = 'dataset_csv/PANDA/PANDA.csv'
    root_path = 'GigaPath_PANDA_embeddings/h5_files' # Or wherever your feature files are
    task_cfg_path = 'finetune/task_configs/panda.yaml'
    save_dir = 'outputs/PANDA' # Base save directory
    exp_name = 'panda_prediction_run' # Experiment name for prediction results

    predict(checkpoint_path, dataset_csv, root_path, task_cfg_path, save_dir, exp_name)
    end_time = time.time()   # 获取代码结束时的时间戳
    elapsed_time = end_time - start_time  # 计算时间差 (秒)
    
    print(f"代码运行时间: {elapsed_time:.4f} 秒") # 格式化输出，保留 4 位小数