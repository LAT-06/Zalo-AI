# QUICK START GUIDE - AeroEyes Challenge

## Setup

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python test_setup.py
```

## Quick Training Example (30 minutes on GPU)

### Option 1: Full Pipeline (Recommended)
```bash
# Run everything with one command
python pipeline.py \
    --train_dir train \
    --test_dir public_test \
    --output_dir output \
    --img_size 640 \
    --epochs 50 \
    --batch 16
```

### Option 2: Step-by-Step

#### Step 1: Parse Dataset (2 minutes)
```bash
python parse_dataset.py \
    --train_dir train \
    --output_dir dataset_yolo \
    --val_split 0.2
```

Expected output:
```
Found 14 classes: ['Backpack', 'Jacket', 'Laptop', ...]
Total videos: 28
Train videos: 22
Val videos: 6
```

#### Step 2: Train Model (20-30 minutes)
```bash
python train_yolov8.py \
    --data dataset_yolo/data.yaml \
    --img_size 640 \
    --epochs 50 \
    --batch 16 \
    --model n \
    --patience 10
```

Training will show:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50      2.5G      1.234      0.567      1.234         48        640
  ...
```

Best model saved at: `runs/train/aeroeyes/weights/best.pt`

#### Step 3: Run Inference on Test Videos (5-10 minutes)
```bash
# Single video
python video_inference.py \
    --model runs/train/aeroeyes/weights/best.pt \
    --video public_test/samples/BlackBox_0/drone_video.mp4 \
    --output results/BlackBox_0.json \
    --conf 0.25

# Or batch process all videos
for video in public_test/samples/*/drone_video.mp4; do
    VIDEO_NAME=$(basename $(dirname "$video"))
    python video_inference.py \
        --model runs/train/aeroeyes/weights/best.pt \
        --video "$video" \
        --output "results/${VIDEO_NAME}.json"
done
```

#### Step 4: Generate Submission (1 minute)
```bash
python generate_submission.py \
    --results_dir results \
    --output submission.json \
    --test_dir public_test/samples
```

Output:
```
Total videos:              6
Videos with detections:    5
Videos without detections: 1
Total bounding boxes:      12345
✓ Submission file generated: submission_20240106_123456.json
```

## Quick Test on Sample Video

```bash
# 1. Download a pre-trained YOLOv8n model (if you don't want to train)
# The model will be downloaded automatically on first use

# 2. Test on single video with visualization
python video_inference.py \
    --model yolov8n.pt \
    --video public_test/samples/BlackBox_0/drone_video.mp4 \
    --output test_result.json \
    --visualize \
    --output_video test_tracked.mp4

# 3. Check the output
cat test_result.json
```

## Validation

```bash
# Evaluate predictions against ground truth
python stiou_metric.py \
    --pred submission.json \
    --gt train/annotations/annotations.json
```

Output:
```
Mean ST-IoU: 0.7234

Per-video scores:
Backpack_0: 0.8123
Laptop_1: 0.7456
...
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_yolov8.py --batch 8 --img_size 512

# Or use CPU (slower)
python train_yolov8.py --device cpu
```

### Low Accuracy
```bash
# Train longer
python train_yolov8.py --epochs 100

# Use larger model
python train_yolov8.py --model s  # or m, l, x
```

### Slow Training
```bash
# Cache images in RAM (requires 8GB+ RAM)
# Modify train_yolov8.py: 'cache': True

# Or reduce image size
python train_yolov8.py --img_size 512
```

## Expected Directory Structure After Setup

```
.
├── train/                          # Training data
│   ├── annotations/
│   │   └── annotations.json
│   └── samples/
│       ├── Backpack_0/
│       │   ├── drone_video.mp4
│       │   └── object_images/
│       └── ...
├── public_test/                    # Test data
│   └── samples/
│       ├── BlackBox_0/
│       └── ...
├── dataset_yolo/                   # Parsed dataset (created by parser)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── runs/                           # Training runs (created by trainer)
│   └── train/
│       └── aeroeyes/
│           └── weights/
│               └── best.pt
├── results/                        # Inference results (created by inference)
│   ├── BlackBox_0.json
│   └── ...
└── submission_*.json              # Final submission file
```

## Time Estimates

- Dataset parsing: 2-5 minutes
- Training (50 epochs, GPU): 20-30 minutes
- Training (100 epochs, GPU): 40-60 minutes
- Inference per video: 30-60 seconds
- Full pipeline: 30-45 minutes

## Next Steps

1. **Improve Performance**:
   - Tune hyperparameters in `train_yolov8.py`
   - Adjust augmentation in `augmentation.py`
   - Try different model sizes (s, m, l)

2. **Deploy to Jetson**:
   - Use `tensorrt_optimize.py` for optimization
   - Build Docker image with provided Dockerfile
   - Deploy with `docker run` command

3. **Analyze Results**:
   - Check per-video ST-IoU scores
   - Visualize predictions with `--visualize` flag
   - Compare different model configurations

## Tips

- Use `--exist_ok` to overwrite previous runs
- Add `--visualize` to see tracking results
- Check tensorboard logs: `tensorboard --logdir runs/train`
- Use `--patience 10` for faster experimentation
- Keep `--conf 0.25` for balanced detection

## Help

If you encounter issues:
1. Run `python test_setup.py` to verify installation
2. Check `README.md` for detailed documentation
3. Review error messages carefully
4. Ensure data structure matches expected format

---
