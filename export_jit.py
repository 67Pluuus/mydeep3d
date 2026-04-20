import torch
from train_cnn import block2CNN, backwarp

def export_to_jit(ckpt_path="../checkpoints/block2_latest.pth", output_path="../checkpoints/block2_1280x720_jit.pt", target_size=(720, 1280)):
    """
    将动态的 .pth 模型权重导出为高性能、固定尺寸的 .pt 静态计算图 (TorchScript JIT)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化原始的动态模型类，并加载 .pth 权重
    model = block2CNN(in_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()  # 必须切换到评估模式，关闭 Dropout/BatchNorm 等训练行为
    
    # 2. 构造一个包含完整推理流程的包裹类 (Wrapper)
    # 因为原始的 __init__ 里只有光流计算，没有加上后期的 Warp 操作
    # 我们希望导出的静态图可以直接吃进左图 [1, 3, H, W]，直接吐出右图预测！
    class FullInferenceWrapper(torch.nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core_model = core_model
            
        def forward(self, left_img):
            # 核心前向：得到光流
            flow = self.core_model(left_img)
            # 外部组合：利用光流扭曲左图
            pred_right = backwarp(left_img, flow)
            return pred_right

    full_model = FullInferenceWrapper(model).to(device)
    full_model.eval()
    
    print(f"准备编译静态图，目标分辨率被硬编码为: {target_size[1]}x{target_size[0]} ...")
    
    # 3. 构造一个 dummy_input (假的数据张量)，形状必须是你未来想要调用的死尺寸！
    h, w = target_size
    dummy_input = torch.randn(1, 3, h, w).to(device)
    
    # 4. 执行 JIT Trace (追踪编译)
    # 这一步 PyTorch 会在底层运行一遍 fake data，记录下所有操作和网格常量
    with torch.no_grad():
        traced_model = torch.jit.trace(full_model, dummy_input)
        
    # 5. 保存为静态的 .pt 文件
    traced_model.save(output_path)
    print(f"太棒了！已成功导出终极加速静态模型至: {output_path}")

if __name__ == "__main__":
    export_to_jit()