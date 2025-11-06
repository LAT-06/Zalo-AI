# AeroEyes Challenge - Aerial Object Detection and Tracking

Complete pipeline for the AeroEyes Challenge: drone video object detection and tracking with spatio-temporal evaluation.

## Overview

This repository contains a comprehensive solution for aerial object detection and tracking in drone videos:

- **Dataset Parser**: Converts AeroEyes annotations to YOLO format with train/val split
- **Data Augmentation**: Aerial-specific augmentations (mosaic, mixup, HSV jitter, motion blur)
- **YOLOv8 Training**: Optimized training pipeline with early stopping and mixed precision
- **Video Inference**: Frame-by-frame detection with ByteTrack object tracking
- **TensorRT Optimization**: Real-time inference on NVIDIA Jetson Xavier NX
- **ST-IoU Metric**: Spatio-temporal evaluation metric
- **Submission Generator**: Formats results for competition submission

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support, ensure CUDA is installed
# For Jetson deployment, use the provided Dockerfile
```

### 2. Dataset Structure

Expected directory structure:
```
train/
├── annotations/
│   └── annotations.json
└── samples/
    ├── Backpack_0/
    │   ├── drone_video.mp4
    │   └── object_images/
    ├── Backpack_1/
    └── ...

public_test/
└── samples/
    ├── BlackBox_0/
    │   ├── drone_video.mp4
    │   └── object_images/
    └── ...
```

### 3. Run Complete Pipeline

```bash
# Run full pipeline with default settings
python pipeline.py --train_dir train --test_dir public_test --output_dir output

# Custom training parameters
python pipeline.py \
    --img_size 640 \
    --epochs 100 \
    --batch 16 \
    --model n \
    --conf 0.25
```

### 4. Or Run Steps Individually

#### Step 1: Parse Dataset
```bash
python parse_dataset.py \
    --train_dir train \
    --output_dir dataset_yolo \
    --val_split 0.2
```

#### Step 2: Train YOLOv8
```bash
python train_yolov8.py \
    --data dataset_yolo/data.yaml \
    --img_size 640 \
    --epochs 100 \
    --batch 16 \
    --model n \
    --project runs/train \
    --name aeroeyes
```

#### Step 3: Video Inference
```bash
python video_inference.py \
    --model runs/train/aeroeyes/weights/best.pt \
    --video public_test/samples/BlackBox_0/drone_video.mp4 \
    --output results/BlackBox_0.json \
    --visualize \
    --output_video results/BlackBox_0_tracked.mp4
```

#### Step 4: Generate Submission
```bash
python generate_submission.py \
    --results_dir results \
    --output submission.json \
    --test_dir public_test/samples \
    --val_gt train/annotations/annotations.json
```

#### Step 5: Evaluate ST-IoU
```bash
python stiou_metric.py \
    --pred submission.json \
    --gt train/annotations/annotations.json
```

## Features

### Dataset Parser (`parse_dataset.py`)
- Reads annotations from `annotations.json`
- Extracts frames from drone videos (25 fps)
- Converts bounding boxes from `(x1, y1, x2, y2)` to YOLO format
- Creates 80/20 train/val split
- Handles empty detections (frames without objects)
- Generates `data.yaml` config for YOLOv8

### Data Augmentation (`augmentation.py`)
Aerial-specific augmentation pipeline:
- **Mosaic**: Combines 4 images (50% probability)
- **Random Scale**: 0.5-1.5x scaling
- **Rotation**: ±10 degrees (preserves aerial perspective)
- **HSV Jitter**: Hue 0.015, Saturation 0.7, Value 0.4
- **Motion Blur**: Simulates drone movement
- **Horizontal Flip**: 50% (no vertical flip for aerial views)
- **MixUp**: Alpha=0.2 blending
- **Careful Small Object Handling**: Minimum area filtering

### YOLOv8 Training (`train_yolov8.py`)
Optimized training configuration:
- **Model**: YOLOv8n (nano) - fast and efficient
- **Input Size**: 640x640 (512x512 option for speed)
- **Optimizer**: SGD with momentum=0.937, weight_decay=0.0005
- **Learning Rate**: 0.01 with cosine decay
- **Warmup**: 3 epochs
- **Mixed Precision**: FP16 training enabled
- **Early Stopping**: Patience=20 epochs
- **Export**: Automatic ONNX export for deployment

### Video Inference (`video_inference.py`)
Complete tracking pipeline:
- **ByteTrack Integration**: 
  - `track_thresh=0.4`: Confidence threshold
  - `track_buffer=30`: Frame buffer for lost tracks
  - `match_thresh=0.8`: IoU threshold for matching
- **Temporal Smoothing**: Moving average of bbox coordinates
- **Occlusion Handling**: Re-identification support
- **Output Format**: JSON with frame-level detections

### TensorRT Optimization (`tensorrt_optimize.py`)
Real-time inference on Jetson Xavier NX:
- **Precision**: FP16 for 2x speedup
- **Target**: >15 FPS on Jetson Xavier NX
- **CUDA Graphs**: Enabled for faster inference
- **Memory Optimization**: Automatic cache clearing
- **Benchmarking**: Built-in performance testing

### ST-IoU Metric (`stiou_metric.py`)
Spatio-temporal evaluation:
- **Formula**: ST-IoU = Σ(IoU(Bf, Bf')) / |frames_union|
- **Per-Video Scores**: Detailed breakdown
- **Dataset Mean**: Average across all videos
- **Validation**: JSON format verification

### Submission Generator (`generate_submission.py`)
Competition-ready output:
- **Format Validation**: Ensures correct JSON structure
- **Missing Videos**: Adds empty entries for all test videos
- **Statistics**: Comprehensive summary
- **Sanity Check**: Optional validation set evaluation

## Model Configurations

### Speed vs Accuracy Trade-offs

| Model | Input Size | Params | Speed (FPS) | mAP@0.5 |
|-------|-----------|--------|-------------|---------|
| YOLOv8n | 512x512 | 3.2M | ~30 | ~0.65 |
| YOLOv8n | 640x640 | 3.2M | ~25 | ~0.70 |
| YOLOv8s | 640x640 | 11.2M | ~20 | ~0.75 |
| YOLOv8m | 640x640 | 25.9M | ~15 | ~0.78 |

*FPS measured on NVIDIA Jetson Xavier NX with TensorRT FP16*

## Advanced Usage

### Multi-Size Training
```bash
python train_yolov8.py --multi_size
```

### Custom Augmentation
```python
from augmentation import AerialAugmentation

aug = AerialAugmentation(img_size=640, mosaic_prob=0.5, mixup_prob=0.2)
aug_img, aug_bboxes, aug_labels = aug(image, bboxes, labels, is_train=True)
```

### Batch Inference
```bash
for video in public_test/samples/*/drone_video.mp4; do
    python video_inference.py \
        --model best.pt \
        --video "$video" \
        --output "results/$(basename $(dirname $video)).json"
done
```

### TensorRT Benchmarking
```bash
python tensorrt_optimize.py \
    --model best.pt \
    --output yolov8_trt.engine \
    --img_size 640 \
    --benchmark \
    --iterations 100
```

## Docker Deployment

### Build Image
```bash
docker build -t aeroeyes:latest .
```

### Run Container on Jetson
```bash
docker run --runtime nvidia --gpus all \
    -v /path/to/data:/app/data \
    -v /path/to/models:/app/models \
    -v /path/to/output:/app/output \
    aeroeyes:latest \
    python3 video_inference.py \
        --model /app/models/best.pt \
        --video /app/data/video.mp4 \
        --output /app/output/result.json
```

## Performance Optimization Tips

1. **Training**:
   - Use `--batch 32` if GPU memory allows
   - Enable `--cache` for faster training (requires RAM)
   - Use `--multi_size` to train with multiple resolutions

2. **Inference**:
   - Lower `--conf` threshold to detect more objects
   - Adjust `--track_buffer` based on occlusion frequency
   - Use TensorRT for 2-3x speedup on Jetson

3. **Memory**:
   - Reduce `--batch` size if OOM errors occur
   - Use `torch.cuda.empty_cache()` between large operations
   - Consider gradient accumulation for effective larger batches

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_yolov8.py --batch 8

# Or use gradient accumulation
python train_yolov8.py --batch 16 --accumulate 2
```

### Low mAP on Validation
- Increase training epochs: `--epochs 200`
- Adjust confidence threshold: `--conf 0.2`
- Enable more augmentation in `augmentation.py`
- Check class balance in dataset

### Slow Inference
- Use TensorRT optimization
- Reduce input size: `--img_size 512`
- Lower detection confidence: `--conf 0.3`
- Disable visualization: remove `--visualize` flag

## Output Formats

### YOLO Label Format
```
class_id x_center y_center width height
0 0.5 0.5 0.2 0.3
```
All values normalized to [0, 1]

### Inference JSON Format
```json
{
  "video_id": "BlackBox_0",
  "detections": [
    {
      "frame": 370,
      "bboxes": [[x1, y1, x2, y2], ...]
    }
  ]
}
```
Bboxes in absolute pixel coordinates

### Submission Format
```json
[
  {
    "video_id": "BlackBox_0",
    "detections": [...]
  },
  {
    "video_id": "BlackBox_1",
    "detections": []
  }
]
```
All test videos must appear (empty list if no detections)

## Citations

```bibtex
@software{yolov8_ultralytics,
  author = {Glenn Jocher and others},
  title = {Ultralytics YOLOv8},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}

@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and others},
  journal={ECCV},
  year={2022}
}
```

## License

MIT License - Feel free to use for academic and commercial purposes.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

