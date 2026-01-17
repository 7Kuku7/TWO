import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import random
from PIL import Image, ImageFilter

# --- 引入依赖 ---
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced

# ==========================================
# 0. 必要的辅助类 (从 train_final.py 复制以解决报错)
# ==========================================
class SelfSupervisedAugmentor:
    def __init__(self):
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        
    def __call__(self, frames):
        # 推理阶段不需要增强，直接返回即可，但保留接口防止报错
        return frames

class AdvancedOFNeRFDataset(OFNeRFDataset):   
    """
    为了兼容 mos_advanced.json (字典格式标签) 而重写的 Dataset
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
        
        # [Fix] 正确解析字典格式的 MOS
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
        
        # 验证/测试模式不做增强
        frames_aug_pil = frames_pil
        
        content_input = self._apply_transform(frames_pil)
        content_input_aug = self._apply_transform(frames_aug_pil)

        # 验证模式不使用 Grid Sampling，直接全图或Resize
        distortion_input = content_input
        distortion_input_aug = content_input_aug
            
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        if not all_frames:
             # 容错处理
             return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]

        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        return [Image.open(p).convert('RGB') for p in selected_frames]

    def _apply_transform(self, pil_list):
        t_imgs = []
        for img in pil_list:
            t_imgs.append(self.transform(img) if self.transform else T.ToTensor()(img))
        return torch.stack(t_imgs)


# ==========================================
# 1. Grad-CAM 核心工具类
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer_type='swin'):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer_type = target_layer_type # 'swin' or 'vit'

    def save_gradient(self, grad):
        self.gradients = grad

    def save_activation(self, module, input, output):
        self.activations = output

    def __call__(self, x_c, x_d, frame_idx=0):
        # 1. Forward Pass
        score, _, _, _, _, _ = self.model(x_c, x_d)
        
        # 2. Backward Pass
        self.model.zero_grad()
        score.backward() 

        # 3. Get Gradients and Activations
        grads = self.gradients
        acts = self.activations
        
        if grads is None or acts is None:
            print("Error: No gradients or activations captured. Check hooks.")
            return np.zeros((224, 224))

        grad = grads[frame_idx].unsqueeze(0) 
        act = acts[frame_idx].unsqueeze(0)
        
        # 4. Generate CAM
        if self.target_layer_type == 'swin':
            if grad.shape[-1] != act.shape[-1]: 
                 weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                 cam = torch.sum(weights * act, dim=1)
            else: 
                 weights = torch.mean(grad, dim=(1, 2), keepdim=True)
                 cam = torch.sum(weights * act, dim=-1)

        elif self.target_layer_type == 'vit':
            num_tokens = act.shape[1]
            grid_size = int(np.sqrt(num_tokens)) 
            if grid_size * grid_size != num_tokens:
                grad = grad[:, 1:, :]
                act = act[:, 1:, :]
                grid_size = int(np.sqrt(num_tokens - 1))
            
            grad = grad.reshape(1, grid_size, grid_size, -1)
            act = act.reshape(1, grid_size, grid_size, -1)
            
            weights = torch.mean(grad, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * act, dim=-1)

        cam = F.relu(cam)
        
        cam = cam.detach().cpu().numpy()[0]
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

# ==========================================
# 2. 注册 Hook 函数
# ==========================================
def register_hooks(model, cam_obj, target_type):
    target_layer = None
    
    if target_type == 'swin':
        # Swin-Tiny: 尝试定位最后的 norm 层
        # 常见路径: model.layers[-1].blocks[-1].norm1
        try:
            target_layer = model.distortion_encoder.layers[-1].blocks[-1].norm1
        except:
            print("Warning: Could not find Swin target layer at default path. Trying alternative...")
            # 备用方案：打印结构并手动调整，或者使用 named_modules 搜索
            pass
            
    elif target_type == 'vit':
        # ViT-Base: model.blocks[-1].norm1
        try:
            target_layer = model.content_encoder.blocks[-1].norm1
        except:
            pass
        
    if target_layer is None:
        print(f"Error: Could not find target layer for {target_type}. Visualization might fail.")
        return

    def forward_hook(module, input, output):
        cam_obj.save_activation(module, input, output)
        
    def backward_hook(module, grad_in, grad_out):
        cam_obj.save_gradient(grad_out[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    print(f"Hook registered on: {target_type}")

# ==========================================
# 3. 可视化主逻辑
# ==========================================
def visualize_sample(model, dataset, index, output_dir, device):
    # 1. 获取数据
    # 现在使用的是 AdvancedOFNeRFDataset，返回 tuple 长度为 7，解包正确
    x_c, x_d, score_gt, _, key, _, _ = dataset[index]
    x_c = x_c.unsqueeze(0).to(device).requires_grad_(True)
    x_d = x_d.unsqueeze(0).to(device).requires_grad_(True)
    
    # 获取原始图片
    frame_tensor = x_c[0, 0].detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = frame_tensor * std + mean
    img_vis = img_vis.permute(1, 2, 0).clamp(0, 1).numpy()
    img_vis = (img_vis * 255).astype(np.uint8)

    # 2. 初始化 CAM
    cam_swin = GradCAM(model, 'swin')
    cam_vit = GradCAM(model, 'vit')
    
    # --- Run for Distortion (Swin) ---
    # 我们重新加载模型以清除 hooks 状态，或者简单点：分别运行
    # 这里采用重新注册的方式
    
    register_hooks(model, cam_swin, 'swin')
    heatmap_d = cam_swin(x_c, x_d, frame_idx=0)
    
    # 清除之前的 hooks 比较麻烦，这里我们利用 python 的机制，
    # 再次注册新的 hooks 到另一个 cam 对象。
    # 为了避免干扰，最好是分别运行。但只要 backward 都能触发就行。
    
    # 注意：为了让 ViT 也能捕捉到梯度，我们需要再次 zero_grad 并 backward
    # 但上面的 cam_swin 已经做了一次 backward。
    # 我们可以复用 score，或者再跑一次 forward。
    # 简单起见：再跑一次 forward
    
    register_hooks(model, cam_vit, 'vit')
    heatmap_c = cam_vit(x_c, x_d, frame_idx=0)

    # 3. 绘制叠加图
    def apply_heatmap(img, heatmap):
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_img = heatmap + np.float32(img) / 255
        cam_img = cam_img / np.max(cam_img)
        return np.uint8(255 * cam_img)

    vis_d = apply_heatmap(img_vis, heatmap_d)
    vis_c = apply_heatmap(img_vis, heatmap_c)

    # 4. 保存
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_vis)
    plt.title(f"Original\nGT: {score_gt.item():.2f}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(vis_d)
    plt.title("Distortion Attention (Swin)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(vis_c)
    plt.title("Content Attention (ViT)")
    plt.axis('off')
    
    # 清理 key 里的路径符号，防止保存出错
    safe_key = str(key).replace('/', '_').replace('\\', '_')
    save_path = os.path.join(output_dir, f"{safe_key}_index{index}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="vis_results/saliency")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--target_name", type=str, default=None, help="Name search (Optional)")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    print("Loading dataset...")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # [Fix] 使用 AdvancedOFNeRFDataset 而不是 OFNeRFDataset
    dataset = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform, num_frames=8)
    
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True)
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 处理索引查找 (保留了 Name Search 功能，方便你以后用)
    target_idx = args.sample_idx
    if args.target_name is not None:
        print(f"Searching for video containing: '{args.target_name}'...")
        for i in range(len(dataset)):
            path_obj = dataset.valid_samples[i]
            if args.target_name in str(path_obj):
                target_idx = i
                print(f"Found match at Index {i}: {path_obj}")
                break

    print(f"Visualizing sample index: {target_idx}")
    visualize_sample(model, dataset, target_idx, args.save_dir, device)
