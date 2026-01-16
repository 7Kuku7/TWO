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
from PIL import Image

# --- 引入依赖 (保持和你项目一致) ---
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced

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
        # output is the activation
        self.activations = output

    def __call__(self, x_c, x_d, frame_idx=0):
        """
        计算特定帧的 Grad-CAM
        frame_idx: 想要可视化这一组输入中的第几帧 (0~T-1)
        """
        # 1. Forward Pass
        # 我们需要保留梯度，所以 model 必须在 train 模式或者开启 requires_grad
        # 但通常 eval 模式下手动开启 input 的 grad 也可以，这里为了简单用 eval + set_grad_enabled
        
        # 这里的 output 是最终的 score
        score, _, _, _, _, _ = self.model(x_c, x_d)
        
        # 2. Backward Pass
        self.model.zero_grad()
        # 我们只关心我们选定的那一帧对分数的贡献，但由于模型是时序平均的，
        # 对 score 求导会自动分发到所有帧。
        score.backward() 

        # 3. Get Gradients and Activations
        # Shapes: [B*T, C, H, W] (for CNN) or [B*T, H, W, C] (Swin) or [B*T, N, C] (ViT)
        grads = self.gradients
        acts = self.activations
        
        # 提取特定帧的数据 (假设 Batch Size = 1)
        # 输入 x_c 是 [1, T, C, H, W]，经过 encoder 后变成 [T, ...]
        # 所以直接按索引取
        grad = grads[frame_idx].unsqueeze(0) # [1, ...]
        act = acts[frame_idx].unsqueeze(0)   # [1, ...]
        
        # 4. Generate CAM
        if self.target_layer_type == 'swin':
            # Swin Output: [1, H, W, C] usually in timm
            # Check shape
            if grad.shape[-1] != act.shape[-1]: # If format matches [1, C, H, W]
                 weights = torch.mean(grad, dim=(2, 3), keepdim=True)
                 cam = torch.sum(weights * act, dim=1)
            else: # Format [1, H, W, C]
                 weights = torch.mean(grad, dim=(1, 2), keepdim=True) # Pool spatial [1, 1, 1, C]
                 cam = torch.sum(weights * act, dim=-1) # [1, H, W]

        elif self.target_layer_type == 'vit':
            # ViT Output: [1, N, C]. N = H*W + 1 (CLS) or just H*W
            # Remove CLS token if present
            num_tokens = act.shape[1]
            grid_size = int(np.sqrt(num_tokens)) 
            if grid_size * grid_size != num_tokens:
                # Assuming first token is CLS
                grad = grad[:, 1:, :]
                act = act[:, 1:, :]
                grid_size = int(np.sqrt(num_tokens - 1))
            
            # Reshape to spatial: [1, H, W, C]
            grad = grad.reshape(1, grid_size, grid_size, -1)
            act = act.reshape(1, grid_size, grid_size, -1)
            
            weights = torch.mean(grad, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * act, dim=-1) # [1, H, W]

        # 5. Post-process (ReLU + Normalize)
        cam = F.relu(cam) # 只看正贡献 (对于分数回归，正贡献=提高分数的区域；或者你可以去掉relu看绝对值)
        # 注意：如果是低分视频，我们可能想看“导致低分”的区域，这时候梯度可能是负的。
        # 建议：可视化 abs(cam) 或者 invert gradients。
        # 简单起见，标准的 Grad-CAM 用 ReLU，表示"Supportive Regions"。
        # 为了看到失真（通常拉低分数），我们可以尝试 score.backward(gradient=torch.tensor(-1.0)) 
        # 但通常直接看 ReLU 也能看到高亮区域，因为模型会聚焦在失真处进行判决。
        
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
        # Swin-Tiny: 通常最后一层在 model.layers[-1].blocks[-1]
        # 需要根据 timm 版本微调，这里是通用猜测
        # DisNeRF 中的 distortion_encoder 是 Swin
        target_layer = model.distortion_encoder.layers[-1].blocks[-1].norm1
    elif target_type == 'vit':
        # ViT-Base: model.blocks[-1].norm1
        target_layer = model.content_encoder.blocks[-1].norm1
        
    if target_layer is None:
        raise ValueError(f"Could not find target layer for {target_type}")

    # Register hooks
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
    x_c, x_d, score_gt, _, key, _, _ = dataset[index]
    x_c = x_c.unsqueeze(0).to(device).requires_grad_(True) # [1, T, C, H, W]
    x_d = x_d.unsqueeze(0).to(device).requires_grad_(True)
    
    # 获取原始图片 (用于叠加) - 取第0帧
    # Dataset 里做了 Normalize，我们需要反归一化方便显示
    frame_tensor = x_c[0, 0].detach().cpu() # [C, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = frame_tensor * std + mean
    img_vis = img_vis.permute(1, 2, 0).clamp(0, 1).numpy()
    img_vis = (img_vis * 255).astype(np.uint8) # [224, 224, 3]

    # 2. 初始化 CAM
    cam_swin = GradCAM(model, 'swin')
    cam_vit = GradCAM(model, 'vit')
    
    # 注册 Hooks (注意：由于是同一个模型，我们需要分别运行两次 forward/backward 或者很小心地清除 hooks)
    # 简单策略：运行两次
    
    # --- Run for Distortion (Swin) ---
    register_hooks(model, cam_swin, 'swin')
    heatmap_d = cam_swin(x_c, x_d, frame_idx=0)
    
    # 清理 hooks 比较麻烦，重新加载模型或者简单的 hack: 
    # 我们这里假设只运行一次 visualize_sample，或者在这里重新实例化模型比较安全，
    # 但为了效率，我们可以在 GradCAM 里仅在 call 时临时 register。
    # (上面的代码为了演示简单，直接注册了。实际运行建议分别跑)
    
    # --- Run for Content (ViT) ---
    # 由于 Hook 已经注册在 Swin 上了，我们再注册一个到 ViT
    # 此时 backward 会同时触发两个 hooks，没问题，因为它们存到不同的 cam 对象里
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
    plt.title("Distortion Attention (Swin)\n(Should highlight Artifacts)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(vis_c)
    plt.title("Content Attention (ViT)\n(Should highlight Objects)")
    plt.axis('off')
    
    save_path = os.path.join(output_dir, f"{key}_vis.png")
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
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load Data & Model
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = OFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform, num_frames=8) # 确保 num_frames 一致
    
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True)
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval() # Eval mode, but we will manually enable gradients for input if needed, though here we use backward on score.

    # 选择几个样本进行可视化
    # 建议手动挑选：一个低分(伪影多)，一个高分(清晰)
    # 这里默认跑参数指定的 sample_idx
    print(f"Visualizing sample index: {args.sample_idx}")
    visualize_sample(model, dataset, args.sample_idx, args.save_dir, device)
