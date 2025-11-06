"""
AeroEyes Dataset Parser
Reads annotations.json and converts to YOLO format with train/val split
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import shutil


class AeroEyesParser:
    def __init__(self, train_dir, output_dir, val_split=0.2):
        """
        Initialize the parser
        
        Args:
            train_dir: Path to train directory containing samples/ and annotations/
            output_dir: Output directory for YOLO format dataset
            val_split: Validation split ratio (default 0.2 for 80/20 split)
        """
        self.train_dir = Path(train_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        
        # Load annotations
        self.annotations_file = self.train_dir / "annotations" / "annotations.json"
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Extract class names from video_ids
        self.class_names = self._extract_class_names()
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        print(f"Total videos: {len(self.annotations)}")
    
    def _extract_class_names(self):
        """Extract unique class names from video IDs"""
        classes = set()
        for video in self.annotations:
            video_id = video['video_id']
            # Extract class name (remove _0, _1 suffix)
            class_name = '_'.join(video_id.split('_')[:-1])
            classes.add(class_name)
        
        # Sort for consistent ordering
        return sorted(list(classes))
    
    def _get_video_info(self, video_path):
        """Get video dimensions and frame count"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        return width, height, frame_count, fps
    
    def _bbox_to_yolo(self, x1, y1, x2, y2, img_width, img_height):
        """
        Convert bbox from (x1, y1, x2, y2) to YOLO format
        
        YOLO format: class_id x_center y_center width height (all normalized 0-1)
        """
        # Calculate center and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize by image dimensions
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # Clip to [0, 1] range
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        return x_center, y_center, width, height
    
    def extract_frames_and_labels(self, video_data, split='train'):
        """
        Extract frames from video and create YOLO labels
        
        Args:
            video_data: Video annotation data
            split: 'train' or 'val'
        """
        video_id = video_data['video_id']
        video_path = self.train_dir / "samples" / video_id / "drone_video.mp4"
        
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            return
        
        # Get video info
        width, height, frame_count, fps = self._get_video_info(video_path)
        print(f"Processing {video_id}: {width}x{height}, {frame_count} frames, {fps:.2f} fps")
        
        # Extract class name
        class_name = '_'.join(video_id.split('_')[:-1])
        class_id = self.class_to_id[class_name]
        
        # Create output directories
        images_dir = self.output_dir / "images" / split
        labels_dir = self.output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Build frame to bboxes mapping
        frame_bboxes = {}
        for ann_obj in video_data.get('annotations', []):
            for bbox in ann_obj.get('bboxes', []):
                frame_num = bbox['frame']
                if frame_num not in frame_bboxes:
                    frame_bboxes[frame_num] = []
                frame_bboxes[frame_num].append({
                    'x1': bbox['x1'],
                    'y1': bbox['y1'],
                    'x2': bbox['x2'],
                    'y2': bbox['y2'],
                    'class_id': class_id
                })
        
        # Open video for frame extraction
        cap = cv2.VideoCapture(str(video_path))
        
        # Sample frames (every 25 frames = 1 second at 25fps)
        # To avoid too many frames, sample strategically
        frames_to_extract = set(frame_bboxes.keys())
        
        # Also sample some frames without objects (for hard negatives)
        all_frames = set(range(0, frame_count, 50))  # Every 2 seconds
        frames_to_extract.update(all_frames)
        
        frame_idx = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frames_to_extract:
                # Save image
                image_filename = f"{video_id}_frame_{frame_idx:06d}.jpg"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), frame)
                
                # Create YOLO label file
                label_filename = f"{video_id}_frame_{frame_idx:06d}.txt"
                label_path = labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    if frame_idx in frame_bboxes:
                        # Frame has objects
                        for bbox in frame_bboxes[frame_idx]:
                            x_c, y_c, w, h = self._bbox_to_yolo(
                                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                                width, height
                            )
                            f.write(f"{bbox['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    # else: empty file for frames with no objects
                
                extracted_count += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"  Extracted {extracted_count} frames from {video_id}")
    
    def parse_and_split(self):
        """Parse entire dataset and create train/val split"""
        # Split videos into train/val
        video_ids = [v['video_id'] for v in self.annotations]
        
        # Stratified split by class
        classes = ['_'.join(vid.split('_')[:-1]) for vid in video_ids]
        
        # Check if we have enough samples for stratified split
        unique_classes = len(set(classes))
        n_samples = len(self.annotations)
        n_test = int(n_samples * self.val_split)
        
        # If test size would be less than number of classes, use regular split
        if n_test < unique_classes:
            print(f"Warning: Too few samples ({n_samples}) for stratified split with {unique_classes} classes")
            print(f"Using random split instead (no stratification)")
            train_videos, val_videos = train_test_split(
                self.annotations,
                test_size=self.val_split,
                random_state=42,
                stratify=None  # Disable stratification
            )
        else:
            train_videos, val_videos = train_test_split(
                self.annotations,
                test_size=self.val_split,
                random_state=42,
                stratify=classes
            )
        
        print(f"\nTrain videos: {len(train_videos)}")
        print(f"Val videos: {len(val_videos)}")
        
        # Process training videos
        print("\n=== Processing Training Set ===")
        for video_data in train_videos:
            self.extract_frames_and_labels(video_data, split='train')
        
        # Process validation videos
        print("\n=== Processing Validation Set ===")
        for video_data in val_videos:
            self.extract_frames_and_labels(video_data, split='val')
        
        # Create data.yaml for YOLOv8
        self.create_data_yaml()
        
        print("\n=== Dataset parsing complete! ===")
        print(f"Output directory: {self.output_dir}")
    
    def create_data_yaml(self):
        """Create data.yaml configuration file for YOLOv8"""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': '',  # No test set with labels
            
            # Number of classes
            'nc': len(self.class_names),
            
            # Class names
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nCreated data.yaml at {yaml_path}")
        print(f"Classes: {self.class_names}")


def main():
    """Main function to parse the dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse AeroEyes dataset to YOLO format')
    parser.add_argument('--train_dir', type=str, default='train',
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='dataset_yolo',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Parse dataset
    parser = AeroEyesParser(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        val_split=args.val_split
    )
    
    parser.parse_and_split()


if __name__ == "__main__":
    main()
