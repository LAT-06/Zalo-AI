#!/bin/bash

# Complete Pipeline Runner for AeroEyes Challenge
# Usage: bash run_complete_pipeline.sh

set -e  # Exit on error

PROJECT_DIR="/home/lat/Documents/ZaloAI/Chal1"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "AEROEYES CHALLENGE - COMPLETE PIPELINE"
echo "========================================================================"
echo ""

# Check Python environment
echo "1️⃣  Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Check if ultralytics is installed
echo ""
echo "2️⃣  Checking required packages..."
if python -c "import ultralytics" 2>/dev/null; then
    echo "✓ ultralytics installed"
else
    echo "⚠ ultralytics not found, installing..."
    pip install ultralytics
fi

# Parse dataset
echo ""
echo "3️⃣  Parsing dataset to YOLO format..."
echo "========================================================================"
if [ -d "dataset_yolo" ]; then
    echo "⚠ dataset_yolo already exists"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf dataset_yolo
        python parse_dataset.py
    else
        echo "Skipping dataset parsing"
    fi
else
    python parse_dataset.py
fi

# Check data.yaml
echo ""
if [ -f "dataset_yolo/data.yaml" ]; then
    echo "✓ data.yaml created successfully"
    echo "Content:"
    cat dataset_yolo/data.yaml
else
    echo "❌ Failed to create data.yaml"
    exit 1
fi

# Train YOLOv8
echo ""
echo "4️⃣  Training YOLOv8 model..."
echo "========================================================================"
python train_yolov8.py \
    --data dataset_yolo/data.yaml \
    --epochs 100 \
    --batch 16 \
    --img_size 640 \
    --patience 20

# Check trained model
echo ""
if [ -f "runs/train/aeroeyes/weights/best.pt" ]; then
    echo "✓ Model trained successfully: runs/train/aeroeyes/weights/best.pt"
else
    echo "❌ Model training failed"
    exit 1
fi

# Test on single video
echo ""
echo "5️⃣  Testing on BlackBox_1 video..."
echo "========================================================================"
python test_single_video.py

# Show results
echo ""
echo "6️⃣  Results Summary"
echo "========================================================================"
echo "✓ Pipeline completed successfully!"
echo ""
echo "Output directories:"
echo "  - Training: runs/train/aeroeyes/"
echo "  - Test: test_output/BlackBox_1/"
echo ""
echo "Next steps:"
echo "  1. Review training results: tensorboard --logdir runs/train"
echo "  2. Test on more videos: python video_inference.py"
echo "  3. Generate submission: python generate_submission.py"
echo ""
echo "========================================================================"
