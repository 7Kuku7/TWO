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
# 假设你的目录结构保持不变
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced

# ==========================================
# 1. 必要的类定义 (复制自 train_final.py 以确保兼容)
# ==========================================
class SelfSupervisedAugmentor:
    def __init__(self):
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    def __call__(self, frames):
        return frames # Visualization 阶段不需要实际增强，占位即可

class AdvancedOFNeRFDataset(OFNeRFDataset):   
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False, enable_ssl=False):
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
        else:
            score = entry / 100.0
        
        frames_pil = self._load_frames_pil(folder_path)
        content_input = self._apply_transform(frames_pil)
        
        # 为了可视化，Distortion Input 我们直接复用 Content Input (无 crop)
        # 或者如果你想严格一致，可以保持原逻辑，这里简化处理
        distortion_input = content_input
            
        # 返回 key 用于解析场景和失真信息
        return content_input, distortion_input, torch.tensor(score, dtype=torch.float32), key

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: raise ValueError(f"No frames found in {folder_path}")
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        return [Image.open(all_frames[i]).convert('RGB') for p in indices]

    def _apply_transform(self, pil_list):
        t_imgs = []
        for img in pil_list:
            t_imgs.append(self.transform(img) if self.transform else T.ToTensor()(img))
        return torch.stack(t_imgs)

# ==========================================
# 2. 核心功能：解析文件名获取标签
# ==========================================
def parse_info_from_key(key):
    """
    根据你的数据集文件名格式，解析出 'Scene' (场景) 和 'Distortion' (失真类型)。
    你需要根据实际情况修改这里的逻辑！
    假设格式如: "lego_fog_level1" 或 "ship_blur_0"
    """
    # [用户请注意]：这里需要根据你的实际文件名格式进行修改
    # 示例逻辑：假设第一个下划线前是场景，中间是失真类型
    parts = key.split('_')
    
    if len(parts) >= 2:
        scene = parts[0]       # 例如 lego
        distortion = parts[1]  # 例如 fog
    else:
        scene = key
        distortion = "unknown"
        
    return scene, distortion

# ==========================================
# 3. 提取特征
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
        for x_c, x_d, score, keys in tqdm(dataloader):
            x_c = x_c.to(device)
            x_d = x_d.to(device)
            
            # Forward Pass
            # model 返回: score, sub_scores, proj_c, proj_d, feat_c, feat_d
            _, _, _, _, feat_c, feat_d = model(x_c, x_d)
            
            feat_c_list.append(feat_c.cpu().numpy())
            feat_d_list.append(feat_d.cpu().numpy())
            mos_scores.extend(score.numpy())
            
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
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    if is_continuous:
        # 如果是连续值 (如 MOS 分数)，用 scatter 颜色映射
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                              c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=legend_title)
    else:
        # 如果是离散值 (如 Scene 名称)，用 seaborn
        unique_labels = np.unique(labels)
        # 如果类别太多，图例会很乱，限制一下
        if len(unique_labels) > 20:
            print(f"Warning: Too many labels ({len(unique_labels)}) for legend.")
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
    # 路径参数
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders", help="数据集路径")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json", help="MOS 文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型 .pth 文件路径")
    parser.add_argument("--save_dir", type=str, default="vis_results", help="结果保存目录")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1. Load Dataset
    print("Loading Dataset...")
    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 这里我们只用验证集或测试集来做可视化
    val_set = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

    # 2. Load Model
    print(f"Loading Model from {args.checkpoint}...")
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True) # 确保参数与训练时一致
    
    # 处理 DataParallel 的 state_dict (如果训练用了多卡，key会有 'module.' 前缀)
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
    feat_c, feat_d, scenes, distortions, mos = extract_features(model, val_loader, device)
    
    print(f"Features extracted: Content {feat_c.shape}, Distortion {feat_d.shape}")

    # 4. Generate Plots
    # Plot A: Content Features (应该按 Scene 聚类)
    plot_tsne(feat_c, scenes, 
              title="Content Feature Space (Colored by Scene)", 
              save_path=os.path.join(args.save_dir, "tsne_content_scene.png"),
              legend_title="Scene")

    # Plot B: Distortion Features (应该按 MOS 或 Distortion Type 聚类)
    # 如果 distortion 名字不统一，用 MOS 着色可能效果更好
    plot_tsne(feat_d, mos, 
              title="Distortion Feature Space (Colored by MOS)", 
              save_path=os.path.join(args.save_dir, "tsne_distortion_mos.png"),
              legend_title="MOS", is_continuous=True)
    
    # Plot C: Distortion Features (Colored by Distortion Type)
    # 只有当你的 distortion 标签比较干净时才有用
    plot_tsne(feat_d, distortions, 
              title="Distortion Feature Space (Colored by Type)", 
              save_path=os.path.join(args.save_dir, "tsne_distortion_type.png"),
              legend_title="Distortion Type")

    print("\nVisualization Finished! Check the 'vis_results' folder.")

if __name__ == "__main__":
    main()
