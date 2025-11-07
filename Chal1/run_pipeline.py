"""
Complete Pipeline Runner for AeroEyes Challenge
Runs: Parse Dataset → Train YOLOv8 → Test on Single Video
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f">>> {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        return False
    
    print(f"\n✓ {description} completed successfully!")
    return True


def main():
    """Main pipeline runner"""
    
    project_dir = Path('/home/lat/Documents/ZaloAI/Chal1')
    os.chdir(project_dir)
    
    print("\n" + "="*70)
    print("AEROEYES CHALLENGE - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 0: Check environment
    print("\n✓ Project directory: " + str(project_dir))
    print(f"✓ Current working directory: {os.getcwd()}")
    
    # Step 1: Check packages
    print("\n1️⃣  Checking required packages...")
    print("="*70)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics/YOLOv8 installed")
    except ImportError:
        print("⚠ ultralytics not found, installing...")
        if not run_command("pip install ultralytics", "Install ultralytics"):
            return False
    
    # Step 2: Parse dataset
    print("\n2️⃣  Parsing dataset to YOLO format...")
    print("="*70)
    
    dataset_yolo = project_dir / "dataset_yolo"
    if dataset_yolo.exists():
        print(f"⚠ {dataset_yolo} already exists")
        response = input("Overwrite? (y/n): ").lower().strip()
        if response == 'y':
            print(f"Removing {dataset_yolo}...")
            shutil.rmtree(dataset_yolo)
            if not run_command("python parse_dataset.py", "Parse dataset"):
                return False
        else:
            print("Skipping dataset parsing")
    else:
        if not run_command("python parse_dataset.py", "Parse dataset"):
            return False
    
    # Check data.yaml
    data_yaml = project_dir / "dataset_yolo" / "data.yaml"
    if not data_yaml.exists():
        print(f"\n❌ Failed to create {data_yaml}")
        return False
    
    print(f"\n✓ {data_yaml} created successfully")
    print("\nContent of data.yaml:")
    with open(data_yaml, 'r') as f:
        print("  " + "\n  ".join(f.read().split('\n')))
    
    # Step 3: Train YOLOv8
    print("\n3️⃣  Training YOLOv8 model on AeroEyes dataset...")
    print("="*70)
    
    train_cmd = (
        "python train_yolov8.py "
        "--data dataset_yolo/data.yaml "
        "--epochs 100 "
        "--batch 16 "
        "--img_size 640 "
        "--patience 20"
    )
    
    if not run_command(train_cmd, "Train YOLOv8"):
        return False
    
    # Check trained model
    best_model = project_dir / "runs" / "train" / "aeroeyes" / "weights" / "best.pt"
    if not best_model.exists():
        print(f"\n❌ Trained model not found: {best_model}")
        return False
    
    print(f"\n✓ Model trained: {best_model}")
    print(f"  Size: {best_model.stat().st_size / (1024*1024):.2f} MB")
    
    # Step 4: Test on single video
    print("\n4️⃣  Testing inference on BlackBox_1 video...")
    print("="*70)
    
    if not run_command("python test_single_video.py", "Test single video"):
        return False
    
    # Step 5: Summary
    print("\n5️⃣  Pipeline Summary")
    print("="*70)
    
    test_output = project_dir / "test_output" / "BlackBox_1"
    if test_output.exists():
        print(f"\n✓ Pipeline completed successfully!")
        print(f"\nKey output files:")
        print(f"  - Model: runs/train/aeroeyes/weights/best.pt")
        print(f"  - Test predictions: {test_output / 'predictions.json'}")
        print(f"  - Test video: {test_output / 'output_tracked.mp4'}")
        print(f"  - Test info: {test_output / 'test_info.json'}")
        
        # Try to print test info
        test_info = test_output / 'test_info.json'
        if test_info.exists():
            import json
            with open(test_info, 'r') as f:
                info = json.load(f)
            print(f"\n✓ Inference Performance:")
            print(f"  - FPS: {info['fps_inference']:.2f}")
            print(f"  - Inference time: {info['inference_time']:.2f}s")
            print(f"  - Total detections: {info['total_detections']}")
            print(f"  - Unique tracks: {info['unique_tracks']}")
    else:
        print(f"\n⚠ Test output directory not created")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run inference on all public_test videos:")
    print("   python video_inference.py --video_dir public_test/samples")
    print("")
    print("2. Generate submission file:")
    print("   python generate_submission.py")
    print("")
    print("3. View training with TensorBoard:")
    print("   tensorboard --logdir runs/train")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
