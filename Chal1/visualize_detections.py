"""
Visualize Detection Results - Draw bounding boxes on frames
"""

import json
import cv2
from pathlib import Path
import argparse


def visualize_detections(video_path, predictions_json, output_dir, sample_frames=None):
    """
    Draw bounding boxes on video frames and save them
    
    Args:
        video_path: Path to input video
        predictions_json: Path to predictions JSON file
        output_dir: Directory to save annotated frames
        sample_frames: List of specific frames to visualize (if None, save all)
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
    
    # Create frame -> bboxes mapping
    frame_to_bboxes = {}
    for detection in predictions['detections']:
        frame_num = detection['frame']
        bboxes = detection['bboxes']
        frame_to_bboxes[frame_num] = bboxes
    
    print(f"✓ Loaded {len(frame_to_bboxes)} frames with detections")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Video: {total_frames} frames @ {fps:.1f} FPS")
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame has detections
        if frame_idx in frame_to_bboxes:
            bboxes = frame_to_bboxes[frame_idx]
            
            # Draw bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Draw label
                text = f"Frame: {frame_idx}"
                cv2.putText(frame, text, (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save frame
            output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Saved {saved_count} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n✓ Visualization complete!")
    print(f"  Saved {saved_count} frames to: {output_dir}")
    print(f"\n  View frames:")
    print(f"    ls -lh {output_dir}/*.jpg")
    print(f"    eog {output_dir}/frame_*.jpg  # Image viewer")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualized frames')
    
    args = parser.parse_args()
    
    visualize_detections(
        video_path=args.video,
        predictions_json=args.predictions,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
