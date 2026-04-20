#!/usr/bin/env python3
import os
import glob
import time
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from pathlib import Path
import logging

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

# 【重点】：这里不再需要 import 任何网络结构代码 (block2CNN) 和 backwarp 函数了！
# 因为所有的运算逻辑和网络参数都已经被死死地“刻”在了 .pt 静态计算图中！

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Metric functions (和 test_cnn.py 中完全一致的严谨评估体系)
# ------------------------------------------------------------------------------
def detect_edges(image, low, high):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, low, high)

def edge_overlap(edge1, edge2):
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0
    return intersection / union

def compute_siou(pred, target, left):
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
    diff = pred.astype(np.float32) - target.astype(np.float32)
    rmse = np.sqrt(np.mean(diff ** 2))
    psnr = 100.0 if rmse == 0 else 20 * np.log10(255.0 / rmse)
    
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
                
    return {'psnr': psnr, 'ssim': ssim_value, 'siou': compute_siou(pred, target, left)}

def convert_crop_and_resize(pil_img, width_and_height):
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
    parser = argparse.ArgumentParser(description="Evaluate Compiled JIT Model")
    parser.add_argument("--jit_ckpt", type=str, default="../checkpoints/block2_1280x720_jit.pt", help="编译好的 JIT 静态图模型路径")
    parser.add_argument("--test_data", type=str, default="../SP_Data/mono2stereo-test", help="测试数据集路径")
    parser.add_argument("--output_dir", type=str, default="../SP_Data/test_results_jit", help="预测输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试数量 (调试用)")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.jit_ckpt):
        logger.error(f"找不到指定的 JIT 模型: {args.jit_ckpt}。请先运行 export_jit.py 生成！")
        return

    # 【超级不同】：直接暴力加载底层的 C++ 计算图！！！
    # 不需要知道任何类名，甚至即使同一目录下没有 train_cnn.py，它也能完美跑起来！
    model = torch.jit.load(args.jit_ckpt, map_location=device)
    model.eval()
    logger.info(f"成功加载最高提速的 JIT 静态计算图模型: {args.jit_ckpt}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    log_file_path = output_root / "jit_evaluation_results.txt"

    subsets = [d for d in os.listdir(args.test_data) if os.path.isdir(os.path.join(args.test_data, d))]
    data_subsets = sorted([s for s in subsets if os.path.exists(os.path.join(args.test_data, s, 'left'))])

    total_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'infer_time': 0, 'count': 0}
    pixel_tf = transforms.Compose([transforms.ToTensor()])

    with open(log_file_path, 'w') as f:
        f.write("JIT Model Evaluation Results\nWeights: {args.jit_ckpt}\n" + "="*80 + "\n")

    for subset in data_subsets:
        subset_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'infer_time': 0.0, 'count': 0}
        left_dir = os.path.join(args.test_data, subset, 'left')
        right_dir = os.path.join(args.test_data, subset, 'right')
        output_subset_dir = output_root / subset
        output_subset_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(left_dir, "*")))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if args.max_samples: image_files = image_files[:args.max_samples]
            
        for img_path in tqdm(image_files, desc=f"Testing {subset}"):
            img_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(img_name)[0]
            
            candidates = glob.glob(os.path.join(right_dir, name_no_ext + ".*"))
            if not candidates: continue
            right_img_path = candidates[0]
                    
            with Image.open(img_path) as pil_left, Image.open(right_img_path) as pil_right:
                # 1. 基准评测面依然保持你的 1280x800
                eval_size = (1280, 800)
                gt_left_eval = convert_crop_and_resize(pil_left, eval_size)
                gt_right_eval = convert_crop_and_resize(pil_right, eval_size)
                
                # 2. 但是送给 JIT 模型前，必须硬性贴合 JIT 编译时“凝固”好的死分辨率尺码！
                model_input_size = (1280, 720) 
                left_input_pil = convert_crop_and_resize(gt_left_eval, model_input_size)
            
            np_left = np.array(gt_left_eval)
            np_right = np.array(gt_right_eval)
            
            left_tf = pixel_tf(left_input_pil).unsqueeze(0).to(device)
            left_tf = (left_tf * 2.0) - 1.0
            
            # 【修复点】：在这里加入了一个强制的 CUDA 同步并预热 GPU
            # 否则有些框架的第一次 CUDA 函数调用会将环境初始化的开销也算进计时器里
            torch.cuda.synchronize()
            
            # 循环跑几次以“平均化”掩盖掉系统极小波动的噪音，提取纯粹的 GPU 计算极速！
            warmup = 2
            for _ in range(warmup):
                _ = model(left_tf)
            torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # ----- 核心推理环节 (超光速！) -----
            with torch.no_grad():
                # 由于我们在 export_jit.py 里包了一层 Wrapper，
                # 所以这个静态图吃进去左图，直接就吐出预测好的右图！不需要在外面算 backwarp 了！
                pred_right_tf = model(left_tf) 
            # -------------------------------
            
            torch.cuda.synchronize()
            infer_time = time.perf_counter() - start_time

            # 3. 将 1280x720 模型输出的右图通过插值拉大到 1280x800 进行指标计算
            pred_right_tf = torch.nn.functional.interpolate(
                pred_right_tf, size=(800, 1280), mode='bilinear', align_corners=False
            )

            pred_right_tf = (pred_right_tf + 1.0) / 2.0
            pred_np = pred_right_tf.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)

            metrics = eval_metrics(pred_np, np_right, np_left)
            
            Image.fromarray(pred_np).save(output_subset_dir / f"{name_no_ext}.png")
            
            subset_metrics['psnr'] += metrics['psnr']
            subset_metrics['ssim'] += metrics['ssim']
            subset_metrics['siou'] += metrics['siou']
            subset_metrics['infer_time'] += infer_time
            subset_metrics['count'] += 1
            
        if subset_metrics['count'] > 0:
            c = subset_metrics['count']
            subset_fps = c / subset_metrics['infer_time']
            msg = (f"Subset: {subset:12s} | Count: {c:4d} | PSNR: {subset_metrics['psnr']/c:6.4f} | "
                   f"SSIM: {subset_metrics['ssim']/c:6.4f} | SIoU: {subset_metrics['siou']/c:6.4f} | "
                   f"FPS: {subset_fps:6.2f}")
            logger.info(msg)
            with open(log_file_path, 'a') as f:
                f.write(msg + "\n")
                
            total_metrics['psnr'] += subset_metrics['psnr']
            total_metrics['ssim'] += subset_metrics['ssim']
            total_metrics['siou'] += subset_metrics['siou']
            total_metrics['infer_time'] += subset_metrics['infer_time']
            total_metrics['count'] += c

    if total_metrics['count'] > 0:
        c = total_metrics['count']
        overall_fps = c / total_metrics['infer_time']
        msg = (f"\n{'='*80}\nOverall Average | Count: {c:4d} | SIoU: {total_metrics['siou']/c:6.4f} | "
               f"SSIM: {total_metrics['ssim']/c:6.4f} | PSNR: {total_metrics['psnr']/c:6.4f} |"
               f"FPS: {overall_fps:6.2f}")
        logger.info(msg)

if __name__ == '__main__':
    main()