#!/bin/bash

# ==============================================================================
# Configuration & Safety Checks (配置与安全检查)
# ==============================================================================

# Stop script immediately on error (遇到错误立即停止)
set -e

# Project Paths (项目路径 - 请确保与您的环境一致)
PROJECT_ROOT="/mnt/c/Users/heror/Desktop/project"
OUTPUT_DIR="$PROJECT_ROOT/data/shapenet/00_good_watertight"
LATEST_RESULT_DIR="$PROJECT_ROOT/_LATEST_EXPERIMENT"

echo "=========================================================="
echo "🚀 NDF-GAN Automated Data Pipeline (全自动数据流水线)"
echo "=========================================================="

# ==============================================================================
# Step 1: Data Selection & Conversion (数据筛选与转换)
# ==============================================================================
# Function: Clears '00_good', picks a random STL, converts to OBJ.
# 功能：清空 '00_good' 文件夹，随机抽取一个 STL 并转换为 OBJ。
echo ""
echo "🎲 [Step 1/4] Random Selection & Conversion (随机抽取并转换)..."
python convert_random_stl.py

# ==============================================================================
# Step 2: Implicit Field Generation (隐式场生成)
# ==============================================================================
# Function: Samples point clouds and calculates SDF/NDF.
# 功能：采样点云并计算 SDF/NDF 距离场。
echo ""
echo "⚡ [Step 2/4] Preprocessing & Sampling (预处理与采样)..."
python prepare_shapenet_dataset.py

# ==============================================================================
# Step 3: Quality Inspection (质量检查)
# ==============================================================================
# Function: Visualizes the generated NDF data to verify sampling quality.
# 功能：可视化生成的 NDF 数据以验证采样质量 (保存为 PNG)。
echo ""
echo "🔍 [Step 3/4] Data Inspection (数据检查)..."
python inspect_data.py

# ==============================================================================
# Step 4: Internal Geometry Augmentation (内部几何增强)
# ==============================================================================
# Function: Generates internal kernels to solve the "hollow object" ambiguity.
# 功能：生成内部内核以解决“空心物体”歧义问题。
echo ""
echo "🧠 [Step 4/4] Generating Internal Structure (生成内部结构)..."
python prepare_watertight_internal_dataset.py

# ==============================================================================
# Post-Processing: Result Management (后处理：结果管理)
# ==============================================================================
# Moves the final result to a separate folder for easy access on Windows.
# 将最终结果移动到单独文件夹，方便在 Windows 上查看。

echo ""
echo "📦 Finalizing: Organizing results (整理结果)..."

# Create/Clear the latest result directory
# 创建或清空最新结果目录
mkdir -p "$LATEST_RESULT_DIR"
rm -rf "$LATEST_RESULT_DIR"/*

# Find the latest generated PLY file from Step 4
# 找到 Step 4 生成的最新的 PLY 文件
LATEST_FILE=$(ls -t "$OUTPUT_DIR"/*/*.ply 2>/dev/null | head -n 1)

if [ -n "$LATEST_FILE" ]; then
    cp "$LATEST_FILE" "$LATEST_RESULT_DIR/"
    echo "✅ Success! The latest model has been copied to:"
    echo "   成功！最新模型已复制到："
    echo "   📂 $LATEST_RESULT_DIR"
    
    # Try to open folder in Windows Explorer (Optional)
    # 尝试在 Windows 资源管理器中打开该文件夹
    explorer.exe $(wslpath -w "$LATEST_RESULT_DIR") 2>/dev/null || true
else
    echo "⚠️ Warning: No output file found in $OUTPUT_DIR"
fi

echo ""
echo "=========================================================="
echo "✅ Pipeline Completed Successfully. (流水线执行完毕)"
echo "=========================================================="