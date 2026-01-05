import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import random
import torchvision.transforms as T
import sys
import os

# Add current directory to path to import consts_simple_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import consts_simple_split as split_utils
except ImportError:
    # Fallback if running from root
    try:
        import TWO.consts_simple_split as split_utils
    except ImportError:
        print("Error: Could not import consts_simple_split")

class OFNeRFDataset(Dataset):
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8):
        """
        Args:
            root_dir: Path to 'renders' directory.
            mos_file: Path to MOS JSON file.
            mode: 'train', 'val', or 'test'.
            transform: Transformations for content branch.
            distortion_sampling: Whether to apply grid mini-patch sampling (for distortion branch).
            num_frames: Number of frames to sample per video.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.distortion_sampling = distortion_sampling
        self.num_frames = num_frames
        
        # Load split
        train, val, test = split_utils.build_simple_split(self.root_dir)
        self.samples = {'train': train, 'val': val, 'test': test}[mode]
        
        # Load MOS labels
        with open(mos_file, 'r') as f:
            self.mos_labels = json.load(f)
            
        # Filter samples that have MOS labels
        self.valid_samples = []
        for s in self.samples:
            key = self._get_key_from_path(s)
            if key in self.mos_labels:
                self.valid_samples.append(s)
            else:
                # Try partial matching or skip
                pass
        
        print(f"[{mode}] Loaded {len(self.valid_samples)} samples with MOS labels.")

    def _get_key_from_path(self, path):
        # Path format: Scene__Method__Condition__Trajectory
        # MOS Key format: Scene+Method+Condition+Trajectory
        parts = path.name.split("__")
        if len(parts) == 4:
            return "+".join(parts)
        return path.name

    def _grid_mini_patch_sampling(self, frames_tensor):
        """
        Apply Grid Mini-Patch Sampling to a tensor of frames [T, C, H, W].
        Splits each frame into 4x4 grid, shuffles patches, and reassembles.
        """
        T, C, H, W = frames_tensor.shape
        # Assuming H, W are divisible by 4 (e.g., 224x224)
        grid_h, grid_w = 4, 4
        patch_h, patch_w = H // grid_h, W // grid_w
        
        # Unfold to patches: [T, C, grid_h, patch_h, grid_w, patch_w]
        patches = frames_tensor.view(T, C, grid_h, patch_h, grid_w, patch_w)
        # Permute to [T, grid_h, grid_w, C, patch_h, patch_w]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Flatten grid: [T, grid_h*grid_w, C, patch_h, patch_w]
        patches = patches.view(T, grid_h * grid_w, C, patch_h, patch_w)
        
        shuffled_patches = []
        for t in range(T):
            # Shuffle indices
            indices = torch.randperm(grid_h * grid_w)
            p = patches[t][indices] # [grid_h*grid_w, C, patch_h, patch_w]
            shuffled_patches.append(p)
            
        shuffled_patches = torch.stack(shuffled_patches) # [T, N_patches, C, pH, pW]
        
        # Reshape back to image
        # [T, grid_h, grid_w, C, pH, pW]
        shuffled_patches = shuffled_patches.view(T, grid_h, grid_w, C, patch_h, patch_w)
        # [T, C, grid_h, pH, grid_w, pW]
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        # [T, C, H, W]
        out = shuffled_patches.view(T, C, H, W)
        
        return out

    def _load_frames(self, folder_path):
        # Load frames
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames:
            raise ValueError(f"No frames found in {folder_path}")
            
        # Uniform sampling
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        
        imgs = []
        for p in selected_frames:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            
        return torch.stack(imgs) # [T, C, H, W]

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        score = self.mos_labels[key]
        
        # Normalize MOS score from [0, 100] to [0, 1]
        score = score / 100.0
        
        frames = self._load_frames(folder_path) # [T, C, H, W] already transformed
        
        content_input = frames
        
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(frames)
        else:
            distortion_input = frames # Fallback or same
            
        return content_input, distortion_input, torch.tensor(score, dtype=torch.float32)

    def __len__(self):
        return len(self.valid_samples)
