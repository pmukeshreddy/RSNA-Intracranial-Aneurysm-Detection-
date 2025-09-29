import torch
import torch.nn as nn
import math

class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=(64, 128, 128), patch_size=(8, 16, 16), in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer3D(nn.Module):
    """3D Vision Transformer for Medical Imaging"""
    def __init__(self, img_size=(64, 128, 128), patch_size=(8, 16, 16), in_channels=3, num_classes=14,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4., qkv_bias=True, drop_rate=0.3):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        
        return x

if __name__ == "__main__":
    # Configuration for Vision Transformer
    config = {
        # Data
        'npz_dir': '/content/drive/MyDrive/kaggle_sep_2025_main/New Folder With Items 2',
        'train_csv': '/content/drive/MyDrive/train.csv',
        'batch_size': 10,  # Smaller batch for ViT (memory intensive)
        'num_workers': 8,
        'fold': 2,  # Train on fold 2 (different from others)
        'use_2d': False,
        'cache_data': False,

        # Model
        'in_channels': 3,
        'num_classes': 14,
        'dropout': 0.3,
        'model_name': 'vit',

        # Training - FAST ENSEMBLE TRAINING
        'learning_rate': 1e-4,  # Lower LR for transformers
        'weight_decay': 5e-5,
        'num_epochs': 5,  # Only 5 epochs
        'patience': 5,
        'use_amp': True,
        'focal_gamma': 2.0,
        'main_label_weight': 2.0,
        'save_dir': './checkpoints_ensemble_vit',
        'use_wandb': False
    }

    print("="*60)
    print("ENSEMBLE MODEL 3: Vision Transformer (ViT)")
    print("="*60)

    # Initialize data module
    data_module = AneurysmDataModule(
        npz_dir=config['npz_dir'],
        train_csv=config['train_csv'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        fold=config['fold'],
        use_2d=config['use_2d'],
        cache_data=config['cache_data']
    )

    # Initialize Vision Transformer model
    model = VisionTransformer3D(
        img_size=(64, 128, 128),
        patch_size=(8, 16, 16),
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=config['dropout']
    )

    # Initialize trainer
    trainer = AneurysmTrainer(model, config, data_module)

    # Train for 5 epochs
    best_score = trainer.train(config['num_epochs'])
    
    print(f"\n✅ Vision Transformer training complete! Best score: {best_score:.4f}")
    print(f"Model saved to: {config['save_dir']}")
