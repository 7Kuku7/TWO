import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import json
import torchvision.transforms as T
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from datetime import datetime
from pathlib import Path

# ==========================================
# 导入你的核心模块
# 确保你已经按照之前的建议建立了 core 文件夹
# ==========================================
try:
    from core.dataset import AdvancedOFNeRFDataset
    from models.dis_nerf_advanced import DisNeRFQA_Advanced
except ImportError:
    print("【错误】找不到 core 模块。请确保你已经按照上一条回答将 train_final.py 拆分为 core/dataset.py 等文件。")
    print("如果尚未拆分，你需要手动将 AdvancedOFNeRFDataset 类定义复制到此文件中。")
    exit(1)

def calculate_metrics(preds, targets):
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    
    srcc = spearmanr(preds, targets)[0]
    plcc = pearsonr(preds, targets)[0]
    krcc = kendalltau(preds, targets)[0]
    rmse = np.sqrt(np.mean((preds - targets)**2))
    
    return srcc, plcc, krcc, rmse

def evaluate_model(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading Split Logic from: consts_simple_split.py (Mode: TEST)")

    # 1. 数据准备 (Standard Resize, No Random Crop for Evaluation)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 关键：mode='test' 会调用 consts_simple_split 获取那 10% 的测试集
    test_set = AdvancedOFNeRFDataset(
        args.root_dir, 
        args.mos_file, 
        mode='test', 
        transform=transform, 
        distortion_sampling=False, # 测试时不做 Grid Sampling，保持原图或 CenterCrop
        use_subscores=args.use_subscores,
        enable_ssl=False # 测试时关闭 SSL 增强
    )
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test Set Size: {len(test_set)} videos")

    # 2. 模型构建
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    
    # 3. 加载权重
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f"Loading checkpoint: {args.checkpoint_path}")
    # 兼容加载整个模型或仅加载 state_dict
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict']) # 某些代码保存的方式
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint) # train_final.py 保存的方式
        
    model.eval()

    # 4. 推理循环
    preds, targets, keys = [], [], []
    
    print("Starting Inference...")
    with torch.no_grad():
        # 注意：dataset 返回 7 个值 (兼容 train_final.py 的逻辑)
        # x_c, x_d, score, sub_scores, key, x_c_aug, x_d_aug
        for batch_data in tqdm(test_loader):
            # 兼容性解包
            if len(batch_data) == 7:
                x_c, x_d, score, _, key, _, _ = batch_data
            else:
                x_c, x_d, score, _, key = batch_data # 旧版本可能返回5个
            
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # 前向传播
            pred_score, _, _, _, _, _ = model(x_c, x_d)
            
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            keys.extend(key)

    # 5. 指标计算 (0-100 分制)
    preds_arr = np.array(preds) * 100.0
    targets_arr = np.array(targets) * 100.0
    
    srcc, plcc, krcc, rmse = calculate_metrics(preds_arr, targets_arr)

    print("\n" + "="*45)
    print(f"   Evaluation Results (Test Set, N={len(preds_arr)})")
    print("="*45)
    print(f"SRCC : {srcc:.4f}")
    print(f"PLCC : {plcc:.4f}")
    print(f"KRCC : {krcc:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("="*45)

    # 6. 保存详细结果 (带时间戳)
    save_dir = "eval_results_standalone"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(args.checkpoint_path).stem
    save_filename = f"Eval_{ckpt_name}_{timestamp}.json"
    save_path = os.path.join(save_dir, save_filename)
    
    per_video_results = []
    for i, k in enumerate(keys):
        per_video_results.append({
            "video": k,
            "pred": float(preds_arr[i]),
            "gt": float(targets_arr[i])
        })
    
    output_data = {
        "meta": {
            "checkpoint": args.checkpoint_path,
            "timestamp": timestamp,
            "dataset_mode": "test",
            "split_source": "consts_simple_split.py"
        },
        "metrics": {
            "SRCC": float(srcc),
            "PLCC": float(plcc),
            "KRCC": float(krcc),
            "RMSE": float(rmse)
        },
        "details": per_video_results
    }
    
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Detailed results saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径修改为你提供的
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    
    # 确保这些参数与训练时一致
    parser.add_argument("--use_subscores", action="store_true", help="Needed if model structure expects subscores")
    parser.add_argument("--no_fusion", action="store_true", help="Set this if you trained with --no_fusion")
    
    args = parser.parse_args()
    evaluate_model(args)
