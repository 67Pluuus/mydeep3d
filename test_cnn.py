#!/usr/bin/env python3
import os
import sys
import glob
import time
import json
import torch
import cv2
import numpy as np
import argparse
import traceback
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from pathlib import Path
import logging

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

# 引入我们刚才在 train_cnn.py 中定义的模型体系和 Warp 函数
from train_cnn import block2CNN, backwarp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Metric functions (从基于 LoRA 的评测文件复制出来的标准代码)
# ------------------------------------------------------------------------------

def detect_edges(image, low, high):
    """Detect edges using Canny edge detector."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low, high)
    return edges

def edge_overlap(edge1, edge2):
    """Calculate edge overlap ratio."""
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0
    return intersection / union

def compute_siou(pred, target, left):
    """Compute Stereo IoU metric."""
    pred_uint8 = pred.astype(np.uint8)
    target_uint8 = target.astype(np.uint8)
    left_uint8 = left.astype(np.uint8)
    
    left_edges = detect_edges(left_uint8, 100, 200)
    pred_edges = detect_edges(pred_uint8, 100, 200)
    right_edges = detect_edges(target_uint8, 100, 200)
    
    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    
    diff_gl = np.abs(pred.astype(np.float32) - left.astype(np.float32))
    diff_rl = np.abs(target.astype(np.float32) - left.astype(np.float32))
    
    if len(diff_gl.shape) == 3:
        diff_gl = cv2.cvtColor(diff_gl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif diff_gl.ndim == 2:
        diff_gl = diff_gl.astype(np.uint8)
    
    if len(diff_rl.shape) == 3:
        diff_rl = cv2.cvtColor(diff_rl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif diff_rl.ndim == 2:
        diff_rl = diff_rl.astype(np.uint8)
    
    diff_gl_ = np.zeros_like(diff_gl)
    diff_rl_ = np.zeros_like(diff_rl)
    diff_gl_[diff_gl > 5] = 1
    diff_rl_[diff_rl > 5] = 1
    
    diff_overlap_grl = edge_overlap(diff_gl_, diff_rl_)
    
    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl

def eval_metrics(pred, target, left):
    """Compute evaluation metrics (PSNR, SSIM, SIoU)."""
    diff = pred.astype(np.float32) - target.astype(np.float32)
    mse_err = np.mean(diff ** 2)
    rmse = np.sqrt(mse_err)
    
    max_pixel = 255.0
    if rmse == 0:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(max_pixel / rmse)
    
    min_dim = min(pred.shape[0], pred.shape[1])
    win_size = 7
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
    
    ssim_value = 0.0
    if win_size >= 3:
        try:
            ssim_ret = ssim(pred, target, full=True, channel_axis=2, win_size=win_size)
            ssim_value = ssim_ret[0] if isinstance(ssim_ret, tuple) else ssim_ret
        except Exception:
            try:
                ssim_ret = ssim(pred, target, full=True, multichannel=True, win_size=win_size)
                ssim_value = ssim_ret[0] if isinstance(ssim_ret, tuple) else ssim_ret
            except Exception:
                pass
                
    siou_value = compute_siou(pred, target, left)
    return {'psnr': psnr, 'ssim': ssim_value, 'siou': siou_value}

def convert_crop_and_resize(pil_img, width_and_height):
    """Convert, crop and resize image (Center Crop)."""
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')
    return ImageOps.fit(pil_img, width_and_height)

# ------------------------------------------------------------------------------
# 主求指标测试例程
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom CNN Model")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/block2_latest.pth", help="训练好的核心块网络权重路径")
    parser.add_argument("--test_data", type=str, default="../SP_Data/mono2stereo-test", help="测试数据集路径")
    parser.add_argument("--output_dir", type=str, default="../SP_Data/test_results_cnn", help="预测输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试数量 (调试用)")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--subsets", type=str, nargs='+', default=None, help="指定要测试的类别 (子文件夹)，例如: animation indoor")
    parser.add_argument("--log_file", type=str, default=None, help="独立日志文件位置")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 实例化模型
    model = block2CNN().to(device)
    model.eval()
    
    # 一个辅助的文件写入器
    def log_to_file(msg):
        if args.log_file:
            with open(args.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
    
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 成功加载权重: {args.ckpt}"
        print(msg, flush=True)
        log_to_file(msg)
    else:
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 警告: 由于 {args.ckpt} 未找到，将使用随机初始化权重跑测试线!"
        print(msg, flush=True)
        log_to_file(msg)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    log_file_path = output_root / "cnn_evaluation_results.txt"
    metrics_file_path = output_root / "cnn_metrics_summary.json"

    # 数据集准备与筛选
    subsets = [d for d in os.listdir(args.test_data) if os.path.isdir(os.path.join(args.test_data, d))]
    data_subsets = []
    for s in subsets:
        if args.subsets and s not in args.subsets:
            continue
        if os.path.exists(os.path.join(args.test_data, s, 'left')) and os.path.exists(os.path.join(args.test_data, s, 'right')):
            data_subsets.append(s)
    data_subsets.sort()

    total_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'infer_time': 0, 'count': 0}
    subset_results = {}

    pixel_tf = transforms.Compose([transforms.ToTensor()])

    with open(log_file_path, 'w') as f:
        f.write("CNN Flow Model Evaluation Results\n")
        f.write(f"Weights: {args.ckpt}\n")
        f.write("=" * 80 + "\n")

    for subset in data_subsets:
        subset_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'infer_time': 0.0, 'count': 0}
        left_dir = os.path.join(args.test_data, subset, 'left')
        right_dir = os.path.join(args.test_data, subset, 'right')
        
        output_subset_dir = output_root / subset
        output_subset_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(left_dir, "*")))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if args.max_samples:
            image_files = image_files[:args.max_samples]
            
        for img_path in tqdm(image_files, desc=f"Testing {subset}"):
            img_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(img_name)[0]
            
            # 由于左右图文件名可以不一致的情况，先精确找，找不到找前缀
            right_img_path = os.path.join(right_dir, img_name)
            if not os.path.exists(right_img_path):
                candidates = glob.glob(os.path.join(right_dir, name_no_ext + ".*"))
                if candidates:
                    right_img_path = candidates[0]
                else:
                    continue
                    
            # 图像加载与对其处理
            with Image.open(img_path) as pil_left, Image.open(right_img_path) as pil_right:
                # 彻底抛弃 1280x800 的畸变过度，严格统一到物理对应的模型尺寸！
                target_size = (1280, 720)  # (width, height)
                left_pil_eval = convert_crop_and_resize(pil_left, target_size)
                right_pil_eval = convert_crop_and_resize(pil_right, target_size)
            
            # 用于最终指标计算的基准图 (1280x720)
            np_left = np.array(left_pil_eval)
            np_right = np.array(right_pil_eval)
            
            # 转Tensor 并缩放到 [-1, 1]，进入网络尺寸为 1280x720
            left_tf = pixel_tf(left_pil_eval).unsqueeze(0).to(device)
            left_tf = (left_tf * 2.0) - 1.0
            
            torch.cuda.synchronize()
            # 预热抗干扰
            for _ in range(2):
                _ = backwarp(left_tf, model(left_tf))
            torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # ----- 核心推理环节 -----
            with torch.no_grad():
                # 输入 1280x720，内部下采样一半变成 640x360 推理，流输出 1280x720
                flow_pred = model(left_tf)
                pred_right_tf = backwarp(left_tf, flow_pred)
            # ------------------------
            
            torch.cuda.synchronize()
            infer_time = time.perf_counter() - start_time

            # 不做任何拉伸！！！原汁原味的 1280x720 进行指标对抗！
            
            # 解析为 uint8 numpy 进行指标计算 (-1~1 -> 0~255)
            pred_right_tf = (pred_right_tf + 1.0) / 2.0
            pred_np = pred_right_tf.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)

            metrics = eval_metrics(pred_np, np_right, np_left)
            
            # 图像保存
            # Image.fromarray(pred_np).save(output_subset_dir / f"{name_no_ext}.png")
            
            subset_metrics['psnr'] += metrics['psnr']
            subset_metrics['ssim'] += metrics['ssim']
            subset_metrics['siou'] += metrics['siou']
            subset_metrics['infer_time'] += infer_time
            subset_metrics['count'] += 1
            
        if subset_metrics['count'] > 0:
            c = subset_metrics['count']
            subset_fps = c / subset_metrics['infer_time']
            msg = (f"Subset: {subset:12s} | Count: {c:4d} | SIoU: {subset_metrics['siou']/c:6.4f} | "
                   f"SSIM: {subset_metrics['ssim']/c:6.4f} | PSNR: {subset_metrics['psnr']/c:6.4f} |"
                   f"FPS: {subset_fps:6.2f}")
            print(msg, flush=True) # 使用 print 并加上 flush 保证缓冲被吐出
            log_to_file(msg)
            
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
                
            subset_results[subset] = {
                'count': c, 'psnr': subset_metrics['psnr']/c, 'ssim': subset_metrics['ssim']/c,
                'siou': subset_metrics['siou']/c, 'fps': subset_fps
            }
                
            total_metrics['psnr'] += subset_metrics['psnr']
            total_metrics['ssim'] += subset_metrics['ssim']
            total_metrics['siou'] += subset_metrics['siou']
            total_metrics['infer_time'] += subset_metrics['infer_time']
            total_metrics['count'] += c

    if total_metrics['count'] > 0:
        c = total_metrics['count']
        overall_fps = c / total_metrics['infer_time']
        msg_header = f"\n{'='*80}\n"
        msg_body = (f"Overall Average | Count: {c:4d} | SIoU: {total_metrics['siou']/c:6.4f} | "
               f"SSIM: {total_metrics['ssim']/c:6.4f} | PSNR: {total_metrics['psnr']/c:6.4f} |"
               f"FPS: {overall_fps:6.2f}")
        print(msg_header + msg_body, flush=True)
        log_to_file(msg_header + msg_body)

if __name__ == '__main__':
    main()
