import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class StereoVideoDataset(Dataset):
    """
    Stereo video dataset for training with robust error handling.
    Loads left/right image pairs and returns N-frame sequences.
    """
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 81,
        image_size: tuple = (1280, 720), # (width, height) - 默认 1280x720 以便对齐部署
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.extensions = extensions
        
        self.left_dir = self.root_dir / "left"
        self.right_dir = self.root_dir / "right"
        
        if not self.left_dir.exists():
            raise FileNotFoundError(f"Left directory not found: {self.left_dir}")
        if not self.right_dir.exists():
            raise FileNotFoundError(f"Right directory not found: {self.right_dir}")
        
        self.left_images = self._find_images(self.left_dir)
        self.right_images = self._find_images(self.right_dir)
        self._validate_image_pairs()
        
        self.num_sequences = len(self.left_images) // self.num_frames
        if self.num_sequences == 0:
            raise ValueError(f"Found {len(self.left_images)} images, need at least {num_frames}.")
        
        logger.info(f"Dataset initialized: {self.num_sequences} sequences")
    
    def _find_images(self, directory: Path) -> list:
        images = []
        for ext in self.extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        
        def extract_number(path):
            import re
            match = re.search(r'\d+', path.stem)
            return int(match.group()) if match else 0
        
        try:
            return sorted(images, key=extract_number)
        except:
            return sorted(images)
    
    def _validate_image_pairs(self):
        min_len = min(len(self.left_images), len(self.right_images))
        self.left_images = self.left_images[:min_len]
        self.right_images = self.right_images[:min_len]
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> dict:
        start_idx = idx * self.num_frames
        if start_idx + self.num_frames > len(self.left_images):
            start_idx = len(self.left_images) - self.num_frames
            
        target_h, target_w = self.image_size[1], self.image_size[0]
        resize_transform = transforms.Resize((target_h, target_w),
            interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            
        left_frames, right_frames = [], []
        
        for i in range(self.num_frames):
            try:
                left_img = Image.open(self.left_images[start_idx + i]).convert('RGB')
                right_img = Image.open(self.right_images[start_idx + i]).convert('RGB')
                
                left_tensor = transforms.ToTensor()(resize_transform(left_img))
                right_tensor = transforms.ToTensor()(resize_transform(right_img))
                
                # Normalize to [-1, 1]
                left_tensor = (left_tensor * 2.0) - 1.0
                right_tensor = (right_tensor * 2.0) - 1.0
                
                left_frames.append(left_tensor)
                right_frames.append(right_tensor)
            except Exception as e:
                dummy = torch.zeros(3, target_h, target_w)
                left_frames.append(dummy.clone())
                right_frames.append(dummy.clone())
                
        return {
            "left": torch.stack(left_frames),
            "right": torch.stack(right_frames)
        }

class block2CNN(nn.Module):
    """
    完全复刻 Deep3D_22/1.py 中 block2 的沙漏型特征提取流程，
    并对标 Deep3D 的推理：预测光流 (Flow)，基于光流使用 grid_sample 对原图进行 Warp 扭曲。
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # conv0: 快速降采样 (1x3xHxW -> 1x48x(H/2)x(W/2) -> 1x96x(H/4)x(W/4))
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.PReLU()
        )
        
        # convblock: 低分辨率下的核心特征融合 (8 组 Conv+PReLU)
        layers = []
        for _ in range(8):
            layers.append(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1))
            layers.append(nn.PReLU())
        self.convblock = nn.Sequential(*layers)
        
        # lastconv: 转置卷积上采样一层 (恢复到 H/2, W/2)
        # Deep3D 中光流只有 u,v 两个通道
        self.lastconv = nn.ConvTranspose2d(96, 2, kernel_size=4, stride=2, padding=1)
        
        # 强制应用双线性插值权重初始化
        self._init_bilinear_weights()

    def _init_bilinear_weights(self):
        """
        对 ConvTranspose2d 的权重进行二维双线性插值核(Bilinear Kernel)初始化。
        消除初始的随机噪点，避免转置卷积前期的“棋盘效应” (Checkerboard Artifacts)。
        """
        w = self.lastconv.weight.data
        in_channels, out_channels, k_h, k_w = w.shape
        
        # 1. 计算双线性插值的基础滤波核矩阵
        factor_h, factor_w = (k_h + 1) // 2, (k_w + 1) // 2
        center_h = factor_h - 0.5 if k_h % 2 == 0 else factor_h - 1
        center_w = factor_w - 0.5 if k_w % 2 == 0 else factor_w - 1
        
        filt = torch.zeros(k_h, k_w)
        for i in range(k_h):
            for j in range(k_w):
                filt[i, j] = (1 - abs(i - center_h) / factor_h) * (1 - abs(j - center_w) / factor_w)
                
        # 2. 覆盖默认随机权重 (因输入有 64 通道，做了一个归一化平均平铺)
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                w[c_in, c_out] = filt / in_channels
                
        if self.lastconv.bias is not None:
            self.lastconv.bias.data.zero_()
            
    def forward(self, x):
        _, _, H, W = x.shape
        
        # 输入端强制下采样 1/2 (输入如 720x1280 -> 360x640)
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # 前置特征提取与降采样 (360x640 -> 180x320 -> 90x160)
        c0 = self.conv0(x_down)
        
        # 深层特征提取 + 残差连接 (Add)
        c_add = c0 + self.convblock(c0)
        
        # 上采样层，输出预测的光流 (Flow) (90x160 -> 180x320)
        flow = self.lastconv(c_add)
        
        # Bilinear Resize: 恢复光流到始端输入的原始大小 (180x320 -> 720x1280)
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        
        # 此时 flow 是基于 180x320 的分辨率尺度预测的，而现在图像拉配到了 HxW (放大了 4 倍)。
        # 因此，为了保证真实的物理像素偏移量对应，光流值必须乘 4.0。
        flow = flow * 4.0
        
        return flow

def backwarp(img, flow):
    """
    完全对标 Deep3D 中的 backwarp 过程，使用 grid_sampler 对图片进行重采样 (Warp)。
    """
    B, C, H, W = img.shape
    
    # 构造标准网格 (Grid) - 采用高性能的 linspace + meshgrid
    # 直接生成 PyTorch 要求的 [-1, 1] 归一化坐标系，省去后续大量的全局除法与减法运算
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, dtype=img.dtype, device=img.device),
        torch.linspace(-1.0, 1.0, W, dtype=img.dtype, device=img.device),
        indexing='ij'
    )
    # 组合为 [H, W, 2] 后，扩展到当前 Batch 的大小: [B, H, W, 2]
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # 将预测的光流通道放到最后: [B, 2, H, W] -> [B, H, W, 2]
    flow = flow.permute(0, 2, 3, 1)
    
    # 将"像素级光流"按比例缩放到 [-1, 1] 的坐标系规范区间中
    # 原始宽度 W 对应区间长度 2，因此每个像素跨度为 2 / (W-1)
    flow_x_norm = flow[..., 0] * (2.0 / max(W - 1, 1))
    flow_y_norm = flow[..., 1] * (2.0 / max(H - 1, 1))
    flow_norm = torch.stack((flow_x_norm, flow_y_norm), dim=-1)
    
    # 最终采样坐标 = 归一化初始网格 + 归一化后的光流偏移量
    vgrid = grid + flow_norm
    
    # 对左图进行重采样 (Warp)，得到右图预测
    output = F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==========================================
    # 1. 完全保留 train_lora.py 的数据读取方法
    # ==========================================
    logger.info(f"Loading dataset from {args.train_dir}...")
    dataset = StereoVideoDataset(
        root_dir=args.train_dir,
        num_frames=args.num_frames,
        image_size=(args.img_width, args.height) # 不写死尺寸，按外部传入处理
    )
    
    # 动态获取 worker 数量，若大于0则允许 persistent_workers
    num_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,                     # 开启锁页内存加速送入GPU
        persistent_workers=(num_workers > 0) # 开启读图进程常驻机制，极大减少每个 epoch 之间的等待时间
    )

    # ==========================================
    # 2. 初始化复刻的 block2 模型及优化器
    # ==========================================
    model = block2CNN(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logger.info(f"Starting training on {device}...")
    logger.info(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ==========================================
    # 3. 基于 Warp 扭曲的光流域训练循环 (Deep3D 流程)
    # ==========================================
    best_loss = float('inf')
    best_model_state = None
    
    # 将进度条提起到最外层，只按 Epoch 更新，不显示多余冗杂信息
    pbar = tqdm(range(args.epochs), desc="Training Process")
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            # 数据打平：[B, T, C, H, W] -> [B*T, C, H, W]
            b, t, c, h, w = batch['left'].shape
            left_img = batch['left'].view(b * t, c, h, w).to(device)
            right_img = batch['right'].view(b * t, c, h, w).to(device)
            
            optimizer.zero_grad()
            
            # Step 1: 网络推理，仅预测光流 (2通道的 (u,v) 偏移场)
            # 在真实的级联推理中，这里可能还会包括上下文特征，但这里是极极简原型。
            flow = model(left_img)
            
            # Step 2: 利用预测的光流场，将左图片 "Warp" (扭曲) 成目标右图片
            # 这里完全对标 `1.py` 中的 w_cur = torch.grid_sampler(input, grid...) 流程！
            pred_right = backwarp(left_img, flow)
            
            # Step 3: 光流自监督计算 Loss，即比对扭曲后的生成结果是否逼近真实右图
            # Deep3D老版本(16年)采用 L1 (MAE) Loss 代替 MSE (L2) 效果更好，边界更锐利
            loss = F.l1_loss(pred_right, right_img)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        # 实时更新最佳分数，并将“候选权重”先极速暂存进内存 (通过 CPU Clone 节省显存并且确保独立性)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # 实时显示当批次指标，此时 best 分数在每一轮都会立马反应最新战绩
        pbar.set_postfix({"avg_mae": f"{avg_loss:.4f}", "best": f"{best_loss if best_loss != float('inf') else 0:.4f}"})
        
        # 4. 控制物理写盘逻辑：每 5 个 epoch，只要内存中积累了全新的最佳权重，才执行一次物理硬盘保存
        if (epoch + 1) % 5 == 0 and best_model_state is not None:
            os.makedirs("../checkpoints", exist_ok=True)
            ckpt_path = f"../checkpoints/block2_latest.pth"
            torch.save(best_model_state, ckpt_path)
            tqdm.write(f"✨ Epoch {epoch+1}: New best model saved to disk -> loss: {best_loss:.4f}")
            best_model_state = None # 落盘后清空待写标记
        
    logger.info(f"Training completed.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight CNN Training based on block2 structure")
    parser.add_argument("--train_dir", type=str, default="../SP_Data/mono_train", help="Path to mono dataset (contains 'left' and 'right' folders)")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/block2_latest.pth", help="Model checkpoint path to save")
    parser.add_argument("--img_width", type=int, default=1280, help="Input image width")
    parser.add_argument("--height", type=int, default=720, help="Input image height")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_frames", type=int, default=1, help="Frames to load per sequence to simulate standard 2D CNN training")
    args = parser.parse_args()
    
    train(args)
