import timm
import torch.nn as nn

def get_content_encoder(pretrained=True):
    """
    ViT-Base for Content Stream.
    Freezes the first 50% of blocks.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
    
    # Freeze first half of blocks
    # ViT-Base has 12 blocks
    blocks = model.blocks
    num_blocks = len(blocks)
    freeze_until = num_blocks // 2
    
    for i in range(freeze_until):
        for param in blocks[i].parameters():
            param.requires_grad = False
            
    # Also freeze patch embed and pos embed usually if we want to keep low-level features stable
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    model.pos_embed.requires_grad = False
    
    return model

def get_distortion_encoder(pretrained=True):
    """
    Swin-Tiny for Distortion Stream.
    """
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=0)
    return model
