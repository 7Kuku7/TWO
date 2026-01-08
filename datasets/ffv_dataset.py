"""
FFV Dataset (Front-Facing Views) for NeRF Quality Assessment
支持 JOD 分数，按 llff/fieldwork/lab 三个子集划分
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import random
import torchvision.transforms as T
import numpy as np


class FFVDataset(Dataset):
    """
    FFV数据集加载器
    
    数据结构:
        benchmark_bank/
        ├── llff/
        │   ├── fern/
        │   │   ├── directvoxgo/
        │   │   │   └── frames/
        │   │   │       ├── frame0000000.jpg
        │   │   │       └── ...
        │   │   └── ...
        │   └── ...
        ├── fieldwork/
        │   └── ...
        └── lab/
            └── ...
    
    标签格式 (JOD分数):
        key: "{subset}+{scene}+{method}"
        例如: "llff+fern+directvoxgo" -> -1.233508549791873
    """
    
    def __init__(self, root_dir, labels_file, subset='all', mode='train', 
                 transform=None, distortion_sampling=False, num_frames=8,
                 train_keys=None, test_keys=None, use_subscores=False):
        """
        Args:
            root_dir: benchmark_bank 根目录路径
            labels_file: 标签JSON文件路径
            subset: 'llff', 'fieldwork', 'lab', 或 'all'
            mode: 'train' 或 'test'
            transform: 图像变换
            distortion_sampling: 是否使用网格patch采样
            num_frames: 每个视频采样的帧数
            train_keys: 训练集的key列表 (从labels_file中读取)
            test_keys: 测试集的key列表 (从labels_file中读取)
            use_subscores: 是否使用子分数 (FFV数据集不支持，保留接口兼容)
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.mode = mode
        self.transform = transform
        self.distortion_sampling = distortion_sampling
        self.num_frames = num_frames
        self.use_subscores = use_subscores
        
        # 加载标签
        with open(labels_file, 'r') as f:
            label_data = json.load(f)
        
        self.all_labels = label_data['labels']
        
        # 获取训练/测试划分
        if train_keys is None:
            train_keys = label_data.get('tr_keys', [])
        if test_keys is None:
            test_keys = label_data.get('tt_keys', [])
        
        # 根据mode选择样本
        if mode == 'train':
            candidate_keys = train_keys
        else:  # val 或 test
            candidate_keys = test_keys
        
        # 根据subset过滤
        if subset != 'all':
            candidate_keys = [k for k in candidate_keys if k.startswith(subset + '+')]
        
        # 验证样本存在性并构建有效样本列表
        self.valid_samples = []
        self.sample_keys = []
        
        for key in candidate_keys:
            parts = key.split('+')
            if len(parts) != 3:
                continue
            subset_name, scene, method = parts
            
            # 构建路径
            sample_path = self.root_dir / subset_name / scene / method / 'frames'
            
            if sample_path.exists() and key in self.all_labels:
                self.valid_samples.append(sample_path)
                self.sample_keys.append(key)
        
        # # 计算JOD分数的归一化参数
        # all_jod_scores = list(self.all_labels.values())
        # self.jod_min = min(all_jod_scores)
        # self.jod_max = max(all_jod_scores)
        
        # print(f"[FFV-{subset}-{mode}] Loaded {len(self.valid_samples)} samples")
        # print(f"  JOD range: [{self.jod_min:.2f}, {self.jod_max:.2f}]")
        
        # 计算JOD分数的归一化参数 (改为 Z-Score 标准化)
        all_jod_scores = list(self.all_labels.values())
        self.jod_mean = np.mean(all_jod_scores)
        self.jod_std = np.std(all_jod_scores)
        
        print(f"[FFV-{subset}-{mode}] Loaded {len(self.valid_samples)} samples")
        print(f"  JOD Mean: {self.jod_mean:.4f}, Std: {self.jod_std:.4f}")
    
    # def _normalize_jod_to_mos(self, jod_score):
    #     """
    #     将JOD分数归一化到[0, 1]范围
    #     JOD分数越高质量越好，归一化后保持这个关系
    #     """
    #     # 线性归一化: (x - min) / (max - min)
    #     normalized = (jod_score - self.jod_min) / (self.jod_max - self.jod_min + 1e-8)
    #     return np.clip(normalized, 0, 1)
    
    def _normalize_jod_to_mos(self, jod_score):
    """
    将JOD分数进行标准化 (Z-Score)
    JOD分数越高质量越好
    """
    # 标准化: (x - mean) / std
    # 结果通常在 -3 到 +3 之间，这让模型更容易区分好坏
    normalized = (jod_score - self.jod_mean) / (self.jod_std + 1e-8)
    return normalized  # 不要 clip，保留真实分布

    def _grid_mini_patch_sampling(self, frames_tensor):
        """
        Grid Mini-Patch Sampling
        将每帧分成4x4网格，随机打乱patch顺序
        """
        T_dim, C, H, W = frames_tensor.shape
        grid_h, grid_w = 4, 4
        patch_h, patch_w = H // grid_h, W // grid_w
        
        patches = frames_tensor.view(T_dim, C, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(T_dim, grid_h * grid_w, C, patch_h, patch_w)
        
        shuffled_patches = []
        for t in range(T_dim):
            indices = torch.randperm(grid_h * grid_w)
            p = patches[t][indices]
            shuffled_patches.append(p)
        
        shuffled_patches = torch.stack(shuffled_patches)
        shuffled_patches = shuffled_patches.view(T_dim, grid_h, grid_w, C, patch_h, patch_w)
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = shuffled_patches.view(T_dim, C, H, W)
        
        return out
    
    def _load_frames(self, folder_path):
        """加载并采样帧"""
        # 支持多种命名格式
        all_frames = sorted(list(folder_path.glob("frame*.jpg")))
        if not all_frames:
            all_frames = sorted(list(folder_path.glob("frame*.png")))
        if not all_frames:
            all_frames = sorted([f for f in folder_path.iterdir() 
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not all_frames:
            raise ValueError(f"No frames found in {folder_path}")
        
        # 均匀采样
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        
        imgs = []
        for p in selected_frames:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            imgs.append(img)
        
        return torch.stack(imgs)
    
    def _load_frames_pil(self, folder_path):
        """加载帧为PIL格式（用于增强）"""
        all_frames = sorted(list(folder_path.glob("frame*.jpg")))
        if not all_frames:
            all_frames = sorted(list(folder_path.glob("frame*.png")))
        if not all_frames:
            all_frames = sorted([f for f in folder_path.iterdir() 
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not all_frames:
            raise ValueError(f"No frames found in {folder_path}")
        
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        
        return [Image.open(p).convert('RGB') for p in selected_frames]
    
    def _apply_transform(self, pil_list):
        """对PIL图像列表应用变换"""
        t_imgs = []
        for img in pil_list:
            if self.transform:
                t_imgs.append(self.transform(img))
            else:
                t_imgs.append(T.ToTensor()(img))
        return torch.stack(t_imgs)
    
    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self.sample_keys[idx]
        
        # 获取JOD分数并归一化
        jod_score = self.all_labels[key]
        normalized_score = self._normalize_jod_to_mos(jod_score)
        score_tensor = torch.tensor(normalized_score, dtype=torch.float32)
        
        # 子分数（FFV不支持，返回零向量保持接口兼容）
        sub_scores_tensor = torch.zeros(4, dtype=torch.float32)
        
        # 加载帧
        frames = self._load_frames(folder_path)
        content_input = frames
        
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(frames)
        else:
            distortion_input = frames
        
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key
    
    def __len__(self):
        return len(self.valid_samples)
    
    def get_subset_from_key(self, key):
        """从key中提取subset名称"""
        return key.split('+')[0]


class AdvancedFFVDataset(FFVDataset):
    """
    带自监督增强的FFV数据集
    兼容 train_final.py 的接口
    """
    
    def __init__(self, root_dir, labels_file, subset='all', mode='train',
                 transform=None, distortion_sampling=False, num_frames=8,
                 train_keys=None, test_keys=None, use_subscores=False):
        super().__init__(root_dir, labels_file, subset, mode, transform,
                        distortion_sampling, num_frames, train_keys, test_keys, use_subscores)
        
        # 导入增强器
        from PIL import ImageFilter
        
        class SelfSupervisedAugmentor:
            def __init__(self):
                self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, 
                                                   saturation=0.3, hue=0.1)
            
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
                if not apply_photo and not apply_geo:
                    apply_photo = True
                
                for img in frames:
                    res = img
                    if apply_geo:
                        res = self.add_geometric_noise(res)
                    if apply_photo:
                        res = self.photo_jitter(res)
                    augmented_frames.append(res)
                return augmented_frames
        
        self.augmentor = SelfSupervisedAugmentor()
    
    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self.sample_keys[idx]
        
        # 获取JOD分数并归一化
        jod_score = self.all_labels[key]
        normalized_score = self._normalize_jod_to_mos(jod_score)
        score_tensor = torch.tensor(normalized_score, dtype=torch.float32)
        
        # 子分数
        sub_scores_tensor = torch.zeros(4, dtype=torch.float32)
        
        # 加载帧（PIL格式用于增强）
        frames_pil = self._load_frames_pil(folder_path)
        
        # 训练时应用增强
        if self.mode == 'train':
            frames_aug_pil = self.augmentor(frames_pil)
        else:
            frames_aug_pil = frames_pil
        
        # 应用变换
        content_input = self._apply_transform(frames_pil)
        content_input_aug = self._apply_transform(frames_aug_pil)
        
        # Distortion采样
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(content_input)
            distortion_input_aug = self._grid_mini_patch_sampling(content_input_aug)
        else:
            distortion_input = content_input
            distortion_input_aug = content_input_aug
        
        return (content_input, distortion_input, score_tensor, sub_scores_tensor, 
                key, content_input_aug, distortion_input_aug)
