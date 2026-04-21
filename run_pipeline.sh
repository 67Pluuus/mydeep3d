#!/bin/bash

# ==========================================
# 华为揭榜 / Deep3D - 端到端训练测试流水线
# ==========================================

# 1. 生成时间戳，格式例如 20260421_153000
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ==========================================
# 自动创建基础大盘目录 (以防整个工程刚刚 Clone 下来)
# ==========================================
mkdir -p ../checkpoints
mkdir -p ../results

# 2. 根据时间戳创建唯一的目标文件夹，用于存放这批跑出来的模型参数
CKPT_DIR="../checkpoints/${TIMESTAMP}"
mkdir -p "${CKPT_DIR}"

# 创建对应的 results 文件夹用于存放 log
RESULTS_DIR="../results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# 训练配置：总轮数、保存频率、学习率
EPOCHS=500   # 总训练轮数
SAVE_FREQ=50 # 填 -1 则只在最后一个 epoch 保存
LR=1e-4      # 学习率

echo ""
echo "=========================================================="
echo " [Stage 1] 启动训练 "
echo " > 模型权重将保存在 -> ${CKPT_DIR} (保存频率: ${SAVE_FREQ}, 学习率: ${LR})"
echo "=========================================================="

# 3. 运行训练代码。保证参数传给 python
python train_cnn.py \
    --ckpt_dir "${CKPT_DIR}" \
    --epochs ${EPOCHS} \
    --save_freq ${SAVE_FREQ} \
    --lr ${LR} \
    --batch_size 4

echo ""
echo "=========================================================="
echo " [Stage 2] 自动化测试评估"
echo "=========================================================="


TEST_SUBSETS=("animation")
# OPTIONAL: "animation" "complex" "indoor" "outdoor" "simple"

# 4. 遍历刚才产生的所有 epoch_xxx.pth，直接套入自动测试
for pth_file in "${CKPT_DIR}"/epoch_*.pth; do
    if [ -f "$pth_file" ]; then
        filename=$(basename "$pth_file" .pth)
        log_file="${RESULTS_DIR}/${filename}.log"
        
        echo ""
        echo ">>>>>>>> 正在评估模型: $pth_file (测试子类: ${TEST_SUBSETS[*]}) <<<<<<<<"
        echo ">>>>>>>> 测试日志正写入: ${log_file} <<<<<<<<"
        
        # 将挑选出的子类变量传入 python 测试脚本
        # 抛弃屏幕 print，将标准输出和错误输出都合并写入对应的 .log 文件
        python test_cnn.py \
            --ckpt "$pth_file" \
            --subsets "${TEST_SUBSETS[@]}" \
            --test_data ../SP_Data/mono2stereo-test \
            --output_dir "../SP_Data/test_results_cnn_${TIMESTAMP}_eval" > "${log_file}"
            
        echo ">>>>>>>> 模型 ${filename} 测试完成！ <<<<<<<<"
    fi
done

echo ""
echo "流水线整体执行完毕！所有结果均已记录。"