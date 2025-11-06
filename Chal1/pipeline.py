"""
End-to-End Pipeline for AeroEyes Challenge
Orchestrates the complete workflow from data parsing to submission generation
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


class AeroEyesPipeline:
    """
    Complete pipeline for the AeroEyes challenge
    """
    
    def __init__(self, train_dir='train', test_dir='public_test', output_dir='output'):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define paths
        self.dataset_dir = self.output_dir / "dataset_yolo"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.submission_dir = self.output_dir / "submissions"
        
        # Create directories
        for d in [self.models_dir, self.results_dir, self.submission_dir]:
            d.mkdir(exist_ok=True)
    
    def run_command(self, command, description):
        """Run a shell command with error handling"""
        print("\n" + "="*70)
        print(f"STEP: {description}")
        print("="*70)
        print(f"Command: {' '.join(command)}")
        print()
        
        result = subprocess.run(command, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: {description} failed with return code {result.returncode}")
            sys.exit(1)
        
        print(f"✓ {description} completed successfully")
    
    def step1_parse_dataset(self, val_split=0.2):
        """Step 1: Parse dataset and create YOLO format"""
        print("\n" + "="*70)
        print("STEP 1: Parsing AeroEyes Dataset")
        print("="*70)
        
        command = [
            sys.executable,
            "parse_dataset.py",
            "--train_dir", str(self.train_dir),
            "--output_dir", str(self.dataset_dir),
            "--val_split", str(val_split)
        ]
        
        self.run_command(command, "Dataset Parsing")
    
    def step2_train_model(self, img_size=640, epochs=100, batch_size=16, model_size='n'):
        """Step 2: Train YOLOv8 model"""
        print("\n" + "="*70)
        print("STEP 2: Training YOLOv8 Model")
        print("="*70)
        
        command = [
            sys.executable,
            "train_yolov8.py",
            "--data", str(self.dataset_dir / "data.yaml"),
            "--img_size", str(img_size),
            "--epochs", str(epochs),
            "--batch", str(batch_size),
            "--model", model_size,
            "--project", str(self.models_dir),
            "--name", "aeroeyes_yolov8",
            "--exist_ok"
        ]
        
        self.run_command(command, "Model Training")
        
        # Find best weights
        weights_path = self.models_dir / "aeroeyes_yolov8" / "weights" / "best.pt"
        if weights_path.exists():
            print(f"✓ Best model saved at: {weights_path}")
            return weights_path
        else:
            print("ERROR: Could not find best.pt weights")
            sys.exit(1)
    
    def step3_optimize_tensorrt(self, model_path, img_size=640):
        """Step 3: Optimize model for TensorRT (optional)"""
        print("\n" + "="*70)
        print("STEP 3: TensorRT Optimization (Optional)")
        print("="*70)
        
        engine_path = self.models_dir / "yolov8_trt.engine"
        
        command = [
            sys.executable,
            "tensorrt_optimize.py",
            "--model", str(model_path),
            "--output", str(engine_path),
            "--img_size", str(img_size),
            "--benchmark"
        ]
        
        try:
            self.run_command(command, "TensorRT Optimization")
            return engine_path
        except Exception as e:
            print(f"WARNING: TensorRT optimization failed: {e}")
            print("Continuing with PyTorch model...")
            return model_path
    
    def step4_inference_test_videos(self, model_path, conf_thresh=0.25):
        """Step 4: Run inference on test videos"""
        print("\n" + "="*70)
        print("STEP 4: Running Inference on Test Videos")
        print("="*70)
        
        # Get all test videos
        test_samples = [d for d in self.test_dir.glob("samples/*") if d.is_dir()]
        
        print(f"Found {len(test_samples)} test videos")
        
        for i, sample_dir in enumerate(test_samples):
            video_id = sample_dir.name
            video_path = sample_dir / "drone_video.mp4"
            
            if not video_path.exists():
                print(f"WARNING: Video not found: {video_path}")
                continue
            
            output_json = self.results_dir / f"{video_id}.json"
            
            print(f"\n[{i+1}/{len(test_samples)}] Processing {video_id}...")
            
            command = [
                sys.executable,
                "video_inference.py",
                "--model", str(model_path),
                "--video", str(video_path),
                "--output", str(output_json),
                "--conf", str(conf_thresh),
                "--track_thresh", "0.4",
                "--track_buffer", "30",
                "--match_thresh", "0.8"
            ]
            
            self.run_command(command, f"Inference on {video_id}")
    
    def step5_generate_submission(self):
        """Step 5: Generate submission file"""
        print("\n" + "="*70)
        print("STEP 5: Generating Submission File")
        print("="*70)
        
        command = [
            sys.executable,
            "generate_submission.py",
            "--results_dir", str(self.results_dir),
            "--output", str(self.submission_dir / "submission.json"),
            "--test_dir", str(self.test_dir / "samples")
        ]
        
        # Add validation GT if available
        val_gt = self.train_dir / "annotations" / "annotations.json"
        if val_gt.exists():
            command.extend(["--val_gt", str(val_gt)])
        
        self.run_command(command, "Submission Generation")
    
    def run_full_pipeline(self, 
                         val_split=0.2,
                         img_size=640,
                         epochs=100,
                         batch_size=16,
                         model_size='n',
                         optimize_tensorrt=False,
                         conf_thresh=0.25):
        """
        Run the complete pipeline
        
        Args:
            val_split: Validation split ratio
            img_size: Input image size
            epochs: Training epochs
            batch_size: Training batch size
            model_size: YOLOv8 model size (n/s/m/l/x)
            optimize_tensorrt: Whether to optimize with TensorRT
            conf_thresh: Confidence threshold for inference
        """
        print("\n" + "="*70)
        print("AEROEYES CHALLENGE - FULL PIPELINE")
        print("="*70)
        print(f"Training directory: {self.train_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
        
        # Step 1: Parse dataset
        self.step1_parse_dataset(val_split=val_split)
        
        # Step 2: Train model
        model_path = self.step2_train_model(
            img_size=img_size,
            epochs=epochs,
            batch_size=batch_size,
            model_size=model_size
        )
        
        # Step 3: Optimize for TensorRT (optional)
        if optimize_tensorrt:
            model_path = self.step3_optimize_tensorrt(model_path, img_size=img_size)
        
        # Step 4: Run inference on test videos
        self.step4_inference_test_videos(model_path, conf_thresh=conf_thresh)
        
        # Step 5: Generate submission
        self.step5_generate_submission()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"Results saved in: {self.output_dir}")
        print(f"Submission file: {self.submission_dir / 'submission_*.json'}")
        print("="*70 + "\n")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline for AeroEyes challenge'
    )
    
    # Directories
    parser.add_argument('--train_dir', type=str, default='train',
                        help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='public_test',
                        help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    
    # Dataset
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Training
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size')
    
    # Optimization
    parser.add_argument('--tensorrt', action='store_true',
                        help='Optimize with TensorRT')
    
    # Inference
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    
    # Pipeline control
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training (use existing model)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (if skip_train)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AeroEyesPipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    if args.skip_train and args.model_path:
        # Skip training, go straight to inference
        print("Skipping training, using provided model...")
        pipeline.step4_inference_test_videos(args.model_path, conf_thresh=args.conf)
        pipeline.step5_generate_submission()
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(
            val_split=args.val_split,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch,
            model_size=args.model,
            optimize_tensorrt=args.tensorrt,
            conf_thresh=args.conf
        )


if __name__ == "__main__":
    main()
