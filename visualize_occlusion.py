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
        # 显式调用父类初始化
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)
    
    def __getitem__(self, idx):
        # 简化版 getitem，只为获取数据
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        # 兼容字典格式的 MOS 标签
        entry = self.mos_labels.get(key)
        # 如果找不到 key，可能因为分割问题，尝试直接用 path.name
        if entry is None:
             entry = self.mos_labels.get(folder_path.name, 0.0)

        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
        else:
            score = entry / 100.0
        
        frames_pil = self._load_frames_pil(folder_path)
        content_input = self._apply_transform(frames_pil)
        
        # Occlusion Map 需要 x_c 和 x_d
        # 这里 x_d 我们也用 content_input (resize后的)，保持一致
        return content_input, content_input, torch.tensor(score), key

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: 
            # 容错：如果是空文件夹
            return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]
            
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
    print(f"Base Score: {base_score:.4f}. Running Occlusion Scan (Patch: {patch_size}, Stride: {stride})...")
    
    y_range = range(0, H - patch_size + 1, stride)
    x_range = range(0, W - patch_size + 1, stride)
    
    # 为了进度条显示，先计算总步数
    total_steps = len(list(y_range)) * len(list(x_range))
    pbar = tqdm(total=total_steps, desc="Scanning", leave=False)

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # 创建遮挡副本
                x_d_occ = x_d.clone()
                # 将该区域设为 0 (黑色)
                x_d_occ[:, :, :, y:y+patch_size, x:x+patch_size] = 0.0 
                
                # 预测分数
                pred_score, _, _, _, _, _ = model(x_c, x_d_occ)
                diff = pred_score.item() - base_score
                
                # 逻辑：
                # diff > 0: 遮挡后分数变高 -> 遮住了伪影 (Bad Region) -> 累加正值
                # diff < 0: 遮挡后分数变低 -> 遮住了细节 (Good Region)
                # 我们主要可视化伪影，但也保留负值以供参考
                
                heatmap[y:y+patch_size, x:x+patch_size] += diff
                counts[y:y+patch_size, x:x+patch_size] += 1
                pbar.update(1)
    
    pbar.close()

    # 平均化
    heatmap = heatmap / (counts + 1e-8)
    
    # 归一化便于显示 (Min-Max Normalize)
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    return heatmap_norm, base_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="vis_results/occlusion")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--target_name", type=str, required=True, help="Partial name or Key of video to visualize")
    # 增加 mode 参数，防止视频在 train 集里导致找不到
    parser.add_argument("--mode", type=str, default="val", choices=['train', 'val', 'test'], help="Which split to search")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading Model from {args.checkpoint}...")
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True)
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    # Load Dataset
    print(f"Loading Dataset ({args.mode})...")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode=args.mode, transform=transform, num_frames=8)
    
    # Find Target
    target_idx = -1
    print(f"Searching for '{args.target_name}' in {len(dataset)} samples...")
    
    for i in range(len(dataset)):
        path_obj = dataset.valid_samples[i]
        
        # 1. 检查 Key (Dataset 生成的 ID，例如 office+mipnerf...)
        key = dataset._get_key_from_path(path_obj)
        
        # 2. 检查 Path String (文件路径，例如 office__mipnerf...)
        path_str = str(path_obj)
        
        # 只要有一个匹配就行
        if (args.target_name in key) or (args.target_name in path_str):
            target_idx = i
            print(f"Found match!\n  - Path: {path_obj}\n  - Key:  {key}")
            break
            
    if target_idx == -1:
        print(f"Error: Video '{args.target_name}' not found in {args.mode} set.")
        print("Tip: If you are sure the video exists, try changing --mode to 'train' or 'test'.")
        return

    # Get Data
    x_c, x_d, score_gt, key = dataset[target_idx]
    x_c = x_c.unsqueeze(0).to(device)
    x_d = x_d.unsqueeze(0).to(device)
    
    # Run Occlusion
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
    
    # 保存结果
    safe_key = str(key).replace('/', '_').replace('\\', '_').replace('+', '_')
    save_path = os.path.join(args.save_dir, f"{safe_key}_occlusion.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Done. Saved to {save_path}")

if __name__ == "__main__":
    main()
