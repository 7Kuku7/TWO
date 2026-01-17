import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import os
import random
from tqdm import tqdm
from PIL import Image, ImageFilter

# --- 引入依赖 ---
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced

# ==========================================
# 1. 必要的类定义
# ==========================================
class SelfSupervisedAugmentor:
    def __init__(self):
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    def __call__(self, frames):
        return frames 

class AdvancedOFNeRFDataset(OFNeRFDataset):   
    """
    独立定义的 Dataset 类，确保返回 4 个值，且兼容字典格式 MOS
    """
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False, enable_ssl=False):
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)
        self.mode = mode
        self.use_subscores = use_subscores
        self.enable_ssl = enable_ssl
        self.augmentor = SelfSupervisedAugmentor()

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        # [关键修复] 兼容字典格式的 MOS 标签
        entry = self.mos_labels.get(key)
        if entry is None: entry = self.mos_labels.get(folder_path.name, 0.0)

        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
        else:
            score = entry / 100.0
        
        frames_pil = self._load_frames_pil(folder_path)
        content_input = self._apply_transform(frames_pil)
        
        # Visualization 只需要 content 和 key
        distortion_input = content_input
            
        # [重点] 必须返回 4 个值，对应 extract_features 的解包
        return content_input, distortion_input, torch.tensor(score, dtype=torch.float32), key

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: 
            return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]

        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_paths = [all_frames[idx] for idx in indices]
        return [Image.open(p).convert('RGB') for p in selected_paths]

    def _apply_transform(self, pil_list):
        t_imgs = []
        for img in pil_list:
            t_imgs.append(self.transform(img) if self.transform else T.ToTensor()(img))
        return torch.stack(t_imgs)

# ==========================================
# 2. 核心功能：解析文件名获取标签
# ==========================================
def parse_info_from_key(key):
    # 简单的 heuristic 解析，建议根据实际情况调整
    parts = key.split('_')
    if len(parts) >= 2:
        scene = parts[0]
        distortion = parts[1]
    elif '+' in key: # 处理 '+' 连接的 key 格式
        parts = key.split('+')
        scene = parts[0]
        distortion = parts[2] if len(parts)>2 else "unknown"
    else:
        scene = key
        distortion = "unknown"
    return scene, distortion

# ==========================================
# 3. 提取特征 (修复变量名问题)
# ==========================================
def extract_features(model, dataloader, device):
    model.eval()
    
    feat_c_list = []
    feat_d_list = []
    scenes = []
    distortions = []
    mos_scores = []
    
    print(">>> Extracting Features...")
    with torch.no_grad():
        # [关键修复] 确保这里的变量名是 keys (复数)，对应 dataloader 返回的 batch key
        for x_c, x_d, score, keys in tqdm(dataloader):
            x_c = x_c.to(device)
            x_d = x_d.to(device)
            
            # Forward Pass
            # model 返回: score, sub_scores, proj_c, proj_d, feat_c, feat_d
            _, _, _, _, feat_c, feat_d = model(x_c, x_d)
            
            feat_c_list.append(feat_c.cpu().numpy())
            feat_d_list.append(feat_d.cpu().numpy())
            mos_scores.extend(score.numpy())
            
            # [关键修复] 这里的 keys 必须与上面循环变量名一致
            for k in keys:
                s, d = parse_info_from_key(k)
                scenes.append(s)
                distortions.append(d)

    return (np.concatenate(feat_c_list, axis=0), 
            np.concatenate(feat_d_list, axis=0), 
            np.array(scenes), 
            np.array(distortions), 
            np.array(mos_scores))

# ==========================================
# 4. t-SNE 绘图
# ==========================================
def plot_tsne(features, labels, title, save_path, legend_title="Class", is_continuous=False):
    print(f"Running t-SNE for {title}...")
    n_samples = features.shape[0]
    # 自动调整 perplexity，防止小样本报错
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    # 如果样本量稍大，强制设为 30 或 50 效果更好；如果很小(如36)，设为 5
    if n_samples < 50: perp = 5
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    if is_continuous:
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                              c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=legend_title)
    else:
        unique_labels = np.unique(labels)
        if len(unique_labels) > 20:
            print(f"Warning: Too many labels ({len(unique_labels)}) for legend. Showing plot without legend.")
            sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                            hue=labels, palette="deep", alpha=0.7, legend=False)
        else:
            sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                            hue=labels, palette="deep", alpha=0.7)
            plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

# ==========================================
# 5. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="vis_results/tsne")
    parser.add_argument("--gpu", type=str, default="0")
    # [新增] 模式选择，建议用 train 获得更多点
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1. Load Dataset
    print(f"Loading Dataset ({args.mode})...")
    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode=args.mode, transform=transform)
    # [关键] drop_last=False 确保所有样本都被处理
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)

    print(f"[{args.mode}] Loaded {len(dataset)} samples.")

    # 2. Load Model
    print(f"Loading Model from {args.checkpoint}...")
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True)
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    # 3. Extract Features
    feat_c, feat_d, scenes, distortions, mos = extract_features(model, dataloader, device)
    
    print(f"Features extracted: Content {feat_c.shape}, Distortion {feat_d.shape}")

    # 4. Generate Plots
    plot_tsne(feat_c, scenes, 
              title=f"Content Feature Space ({args.mode} set)", 
              save_path=os.path.join(args.save_dir, "tsne_content_scene.png"),
              legend_title="Scene")

    plot_tsne(feat_d, mos, 
              title=f"Distortion Feature Space ({args.mode} set - MOS)", 
              save_path=os.path.join(args.save_dir, "tsne_distortion_mos.png"),
              legend_title="MOS", is_continuous=True)
    
    plot_tsne(feat_d, distortions, 
              title=f"Distortion Feature Space ({args.mode} set - Type)", 
              save_path=os.path.join(args.save_dir, "tsne_distortion_type.png"),
              legend_title="Type")

    print("\nVisualization Finished!")

if __name__ == "__main__":
    main()
