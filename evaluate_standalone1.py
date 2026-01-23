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
from PIL import Image, ImageFilter
import random

# ==========================================
# 1. 导入项目依赖
# ==========================================
# 确保 datasets 和 models 文件夹在当前目录下
try:
    from datasets.of_nerf import OFNeRFDataset
    from models.dis_nerf_advanced import DisNeRFQA_Advanced
    # 如果你需要 utils 里的计算函数也可以导入，这里为了独立性直接在下面重写了计算逻辑
except ImportError as e:
    print(f"【环境错误】无法导入基础模块: {e}")
    print("请确保你在项目根目录下运行此脚本，且存在 'datasets' 和 'models' 文件夹。")
    exit(1)

# ==========================================
# 2. 从 train_final.py 复制过来的核心类
#    (因为还没有拆分模块，所以直接在这里定义)
# ==========================================

class SelfSupervisedAugmentor:
    """
    [复制自 train_final.py] 自监督增强模块
    """
    def __init__(self):
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        
    def add_geometric_noise(self, img):
        choice = random.choice(['blur', 'pixelate'])
        if choice == 'blur':
            radius = random.uniform(1, 3)
            return img.filter(ImageFilter.GaussianBlur(radius))
        elif choice == 'pixelate':
            w, h = img.size
            ratio = random.uniform(0.2, 0.5)
            img_small = img.resize((int(w*ratio), int(h*ratio)), resample=Image.NEAREST)
            return img_small.resize((w, h), resample=Image.NEAREST)
        return img

    def __call__(self, frames):
        augmented_frames = []
        apply_photo = random.random() > 0.3
        apply_geo = random.random() > 0.3
        if not apply_photo and not apply_geo: apply_photo = True 
            
        for img in frames:
            res = img
            if apply_geo: res = self.add_geometric_noise(res)
            if apply_photo: res = self.photo_jitter(res)
            augmented_frames.append(res)
        return augmented_frames

class AdvancedOFNeRFDataset(OFNeRFDataset):   
    """
    [复制自 train_final.py] 包含 SSL 和 Sub-scores 的 Dataset
    """
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False, enable_ssl=True):
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)

        self.mode = mode
        self.use_subscores = use_subscores
        self.enable_ssl = enable_ssl
        self.augmentor = SelfSupervisedAugmentor()

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        entry = self.mos_labels[key]
        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
            sub_data = entry.get('sub_scores', {})
        else:
            score = entry / 100.0
            sub_data = {}
        
        score_tensor = torch.tensor(score, dtype=torch.float32)
        sub_scores_tensor = torch.zeros(4, dtype=torch.float32)
        if self.use_subscores:
            sub_scores_tensor = torch.tensor([
                sub_data.get("discomfort", 0), sub_data.get("blur", 0),
                sub_data.get("lighting", 0), sub_data.get("artifacts", 0)
            ], dtype=torch.float32) / 5.0
        
        frames_pil = self._load_frames_pil(folder_path)
        
        # 即使是 test 模式代码逻辑也要保持一致，虽然 enable_ssl=False 时不增强
        if self.mode == 'train' and self.enable_ssl:
            frames_aug_pil = self.augmentor(frames_pil)
        else:
            frames_aug_pil = frames_pil
        
        content_input = self._apply_transform(frames_pil)
        content_input_aug = self._apply_transform(frames_aug_pil)

        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(content_input)
            distortion_input_aug = self._grid_mini_patch_sampling(content_input_aug)
        else:
            distortion_input = content_input
            distortion_input_aug = content_input_aug
            
        # 返回 7 个值
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        if not all_frames: raise ValueError(f"No frames found in {folder_path}")

        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        return [Image.open(p).convert('RGB') for p in selected_frames]

    def _apply_transform(self, pil_list):
        t_imgs = []
        for img in pil_list:
            t_imgs.append(self.transform(img) if self.transform else T.ToTensor()(img))
        return torch.stack(t_imgs)


# ==========================================
# 3. 评估逻辑
# ==========================================

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
    
    # 1. 数据准备 (测试集仅 Resize，不做 RandomCrop)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading Test Set... (Mode: test)")
    test_set = AdvancedOFNeRFDataset(
        args.root_dir, 
        args.mos_file, 
        mode='test',    # 强制使用测试集
        transform=transform, 
        distortion_sampling=False, # 测试时不做 Grid Patch Sampling (或根据你训练时的设置)
        use_subscores=args.use_subscores,
        enable_ssl=False # 评估模式必须关闭 SSL 增强
    )
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test Set Size: {len(test_set)} videos")

    # 2. 模型构建
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    
    # 3. 加载权重
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 兼容 state_dict 或完整 checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 这种通常是你的 train_final.py 直接 save(model.state_dict()) 的情况
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 4. 推理
    preds, targets, keys = [], [], []
    
    print("Starting Inference...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            # 解包：兼容 Dataset 返回的 7 个值
            # content, distortion, score, sub_scores, key, aug_c, aug_d
            if len(batch_data) == 7:
                x_c, x_d, score, _, key, _, _ = batch_data
            else:
                x_c, x_d, score, _, key = batch_data 
            
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # Forward
            pred_score, _, _, _, _, _ = model(x_c, x_d)
            
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            keys.extend(key)

    # 5. 指标计算 (转换为 0-100 分制以便观察 RMSE)
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

    # 6. 保存详细结果
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
            "mode": "test"
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
    # 默认参数 (请根据你的实际路径修改 default)
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the best_model.pth")
    parser.add_argument("--batch_size", type=int, default=1) # 评估时用 1 或小一点没问题
    parser.add_argument("--gpu", type=str, default="0")
    
    # 必须与训练时的参数保持一致，否则权重加载会报错
    parser.add_argument("--use_subscores", action="store_true", help="If model trained with subscores")
    parser.add_argument("--no_fusion", action="store_true", help="If model trained without fusion")
    
    args = parser.parse_args()
    
    evaluate_model(args)
