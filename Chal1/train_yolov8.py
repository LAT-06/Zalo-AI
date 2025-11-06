"""
YOLOv8 Training Script for AeroEyes Dataset
Trains YOLOv8n model with aerial-specific configurations
"""

import os
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import argparse
from datetime import datetime


def train_yolov8(
    data_yaml,
    model_size='n',
    img_size=640,
    batch_size=16,
    epochs=100,
    patience=20,
    device='0',
    project='runs/train',
    name='aeroeyes',
    resume=False,
    exist_ok=False
):
    """
    Train YOLOv8 model on AeroEyes dataset
    
    Args:
        data_yaml: Path to data.yaml config file
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        img_size: Input image size
        batch_size: Batch size for training
        epochs: Number of training epochs
        patience: Early stopping patience
        device: GPU device (e.g., '0' or 'cpu')
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
        exist_ok: Overwrite existing experiment
    """
    
    # Initialize model
    model_name = f'yolov8{model_size}.pt'
    print(f"Initializing {model_name}...")
    model = YOLO(model_name)
    
    # Training hyperparameters (optimized for aerial footage)
    hyperparams = {
        # Data
        'data': data_yaml,
        'imgsz': img_size,
        'batch': batch_size,
        'epochs': epochs,
        'patience': patience,
        
        # Device
        'device': device,
        
        # Optimizer settings
        'optimizer': 'SGD',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation (handled by albumentations, so disable some)
        'hsv_h': 0.015,  # HSV hue augmentation
        'hsv_s': 0.7,    # HSV saturation augmentation
        'hsv_v': 0.4,    # HSV value augmentation
        'degrees': 10.0,  # Rotation degrees
        'translate': 0.1,  # Translation
        'scale': 0.5,      # Scale
        'shear': 0.0,      # Shear
        'perspective': 0.0,  # Perspective
        'flipud': 0.0,      # Flip up-down (disable for aerial)
        'fliplr': 0.5,      # Flip left-right
        'mosaic': 1.0,      # Mosaic augmentation
        'mixup': 0.2,       # MixUp augmentation
        'copy_paste': 0.0,  # Copy-paste augmentation
        
        # Loss weights
        'box': 7.5,      # Box loss weight
        'cls': 0.5,      # Classification loss weight
        'dfl': 1.5,      # Distribution focal loss weight
        
        # Training settings
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic in last N epochs
        
        # Validation
        'val': True,
        'save': True,
        'save_period': -1,  # Save checkpoint every N epochs (-1 = disabled)
        'cache': False,  # Cache images for faster training (use with caution)
        'workers': 8,    # Number of dataloader workers
        
        # Mixed precision
        'amp': True,  # Automatic Mixed Precision (FP16)
        
        # Logging
        'project': project,
        'name': name,
        'exist_ok': exist_ok,
        'verbose': True,
        
        # Visualization
        'plots': True,
        
        # Resume
        'resume': resume,
    }
    
    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    for key, value in hyperparams.items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")
    
    # Train model
    print("Starting training...")
    results = model.train(**hyperparams)
    
    # Validate
    print("\nValidating best model...")
    metrics = model.val()
    
    # Print results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Best mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Best weights saved to: {model.trainer.best}")
    print("="*60 + "\n")
    
    # Export to ONNX for deployment
    print("Exporting to ONNX format...")
    try:
        # Try with simplification first (requires onnxsim)
        try:
            onnx_path = model.export(format='onnx', imgsz=img_size, simplify=True)
            print(f"ONNX model saved to: {onnx_path} (simplified)")
        except:
            # Fallback to export without simplification
            onnx_path = model.export(format='onnx', imgsz=img_size, simplify=False)
            print(f"ONNX model saved to: {onnx_path} (not simplified)")
            print("Note: Install onnxsim for model simplification (requires cmake)")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        print("This is optional - model training completed successfully")
    
    return model, metrics


def train_multi_size():
    """Train models with different input sizes for speed comparison"""
    sizes = [512, 640]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Training with image size: {size}x{size}")
        print(f"{'='*60}\n")
        
        train_yolov8(
            data_yaml='dataset_yolo/data.yaml',
            img_size=size,
            name=f'aeroeyes_{size}',
            exist_ok=True
        )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 on AeroEyes dataset')
    
    # Data
    parser.add_argument('--data', type=str, default='dataset_yolo/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size (default: 640)')
    
    # Model
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    # Training
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (e.g., 0 or 0,1,2,3) or cpu')
    
    # Experiment
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='aeroeyes',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Overwrite existing experiment')
    
    # Multi-size training
    parser.add_argument('--multi_size', action='store_true',
                        help='Train with multiple input sizes (512, 640)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, switching to CPU")
            args.device = 'cpu'
        else:
            print(f"Using CUDA device: {args.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Train
    if args.multi_size:
        train_multi_size()
    else:
        train_yolov8(
            data_yaml=args.data,
            model_size=args.model,
            img_size=args.img_size,
            batch_size=args.batch,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            exist_ok=args.exist_ok
        )


if __name__ == "__main__":
    main()
