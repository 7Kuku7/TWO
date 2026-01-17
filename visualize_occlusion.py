import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

# 引入您的模型和数据集
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced

# 必须包含这个 Dataset wrapper 才能正确读取字典格式 MOS
class AdvancedOFNeRFDataset(OFNeRFDataset):   
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False, enable_ssl=False):
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)
    
    def __getitem__(self, idx):
        # 简化版 getitem，只为获取数据
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        entry = self.mos_labels[key]
        score = entry['mos'] / 100.0 if isinstance(entry, dict) else entry / 100.0
        
        frames_pil = self._load_frames_pil(folder_path)
        content_input = self._apply_transform(frames_pil)
        return content_input, content_input, torch.tensor(score), key

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        return [Image.open(all_frames[i]).convert('RGB') for i in indices]

    def _apply_transform(self, pil_list):
        t_imgs = [self.transform(img) for img in pil_list]
        return torch.stack(t_imgs)

def generate_occlusion_map(model, x_c, x_d, device, patch_size=32, stride=16):
    """
    生成遮挡敏感度图
    """
    model.eval()
    
    # 1. 原始分数
    with torch.no_grad():
        base_score, _, _, _, _, _ = model(x_c, x_d)
        base_score = base_score.item()
    
    B, T, C, H, W = x_c.shape
    # 热力图初始化
    heatmap = np.zeros((H, W))
    counts = np.zeros((H, W))
    
    # 2. 滑动窗口遮挡
    # 我们只遮挡 x_d (Distortion Input) 来查看哪些区域影响了质量判断
    # x_c 保持不变，作为 Context
    
    print(f"Base Score: {base_score:.4f}. Running Occlusion Scan...")
    
    y_range = range(0, H - patch_size + 1, stride)
    x_range = range(0, W - patch_size + 1, stride)
    
    total_steps = len(y_range) * len(x_range)
    
    with torch.no_grad():
        for y in y_range:
            for x in x_range:
                # 创建遮挡副本
                x_d_occ = x_d.clone()
                # 将该区域设为 0 (黑色) 或 均值灰色
                x_d_occ[:, :, :, y:y+patch_size, x:x+patch_size] = 0.0 
                
                # 预测分数
                pred_score, _, _, _, _, _ = model(x_c, x_d_occ)
                diff = pred_score.item() - base_score
                
                # 逻辑：
                # 如果遮挡后分数变高 (diff > 0)，说明遮住的是“坏东西”（伪影）。
                # 如果遮挡后分数变低 (diff < 0)，说明遮住的是“好东西”（细节）。
                # 我们主要想看伪影，所以关注 diff > 0 的区域。
                
                heatmap[y:y+patch_size, x:x+patch_size] += diff
                counts[y:y+patch_size, x:x+patch_size] += 1

    # 平均化
    heatmap = heatmap / (counts + 1e-8)
    
    # 归一化便于显示 (只取正值部分高亮伪影)
    # heatmap = np.maximum(heatmap, 0) 
    # 或者全范围归一化
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    return heatmap, base_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="vis_results/occlusion")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--target_name", type=str, required=True, help="Partial name of video to visualize")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True)
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    # Load Dataset
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform, num_frames=8)
    
    # Find Target
    target_idx = -1
    for i in range(len(dataset)):
        path_obj = dataset.valid_samples[i]
        if args.target_name in str(path_obj):
            target_idx = i
            print(f"Found: {path_obj}")
            break
            
    if target_idx == -1:
        print("Video not found.")
        return

    # Get Data
    x_c, x_d, score_gt, key = dataset[target_idx]
    x_c = x_c.unsqueeze(0).to(device)
    x_d = x_d.unsqueeze(0).to(device)
    
    # Run Occlusion
    print("Generating Occlusion Sensitivity Map (This may take a minute)...")
    heatmap, pred_score = generate_occlusion_map(model, x_c, x_d, device, patch_size=32, stride=8)
    
    # Visualization
    frame_tensor = x_c[0, 0].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = frame_tensor * std + mean
    img_vis = img_vis.permute(1, 2, 0).clamp(0, 1).numpy()
    img_vis = (img_vis * 255).astype(np.uint8)
    
    heatmap_vis = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap_vis, 0.4, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_vis)
    plt.title(f"Original (GT: {score_gt:.2f}, Pred: {pred_score:.2f})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Occlusion Sensitivity\n(Red = Area causing Quality Drop)")
    plt.axis('off')
    
    safe_key = str(key).replace('/', '_')
    plt.savefig(os.path.join(args.save_dir, f"{safe_key}_occlusion.png"), bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    main()
