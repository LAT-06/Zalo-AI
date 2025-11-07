"""
Quick Test Pipeline on Single Video (BlackBox_1)
Test inference and tracking on a single public_test video
"""

import os
import sys
import cv2
import json
import torch
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video_inference import VideoInference


def test_single_video():
    """Test on BlackBox_1 video from public_test"""
    
    print("\n" + "="*70)
    print("SINGLE VIDEO TEST - BlackBox_1")
    print("="*70 + "\n")
    
    # Paths
    video_path = Path('public_test/samples/BlackBox_1/drone_video.mp4')
    output_dir = Path('test_output/BlackBox_1')
    model_path = Path('runs/train/aeroeyes/weights/best.pt')
    
    # Check if video exists
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    print(f"✓ Video found: {video_path}")
    
    # Check if model exists (if not, we'll use pretrained)
    if not model_path.exists():
        print(f"⚠ Trained model not found: {model_path}")
        print("Will use pretrained YOLOv8n model instead")
        model_path = 'yolov8n.pt'
    else:
        print(f"✓ Model found: {model_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f}s\n")
    
    # Run inference
    print("Running inference and tracking...")
    print("-" * 70)
    
    start_time = datetime.now()
    
    # Run inference using VideoInference class
    pipeline = VideoInference(
        model_path=str(model_path),
        conf_thresh=0.25,
        iou_thresh=0.45,
        track_thresh=0.4,
        track_buffer=30,
        match_thresh=0.8,
        smooth_window=5,
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    predictions = pipeline.process_video(
        video_path=str(video_path),
        output_json=str(output_dir / 'predictions.json'),
        visualize=False,
        output_video=str(output_dir / 'output_tracked.mp4')
    )
    
    elapsed_time = datetime.now() - start_time
    
    print("-" * 70)
    print(f"\n✓ Inference completed in {elapsed_time.total_seconds():.2f}s")
    print(f"  Average: {elapsed_time.total_seconds()/total_frames*1000:.2f}ms per frame")
    print(f"  FPS: {total_frames/elapsed_time.total_seconds():.2f}")
    
    # Save predictions (already saved by pipeline)
    print(f"\nPredictions saved by pipeline")
    
    # Statistics from predictions dict
    print(f"\nPrediction Statistics:")
    total_dets = sum(len(frame.get('bboxes', [])) for frame in predictions.get('detections', []))
    unique_tracks = set()
    for frame in predictions.get('detections', []):
        for bbox in frame.get('bboxes', []):
            # bbox format from VideoInference
            if len(bbox) >= 4:
                unique_tracks.add(tuple(bbox[:4]))  # Use bbox as track identifier
    
    print(f"  Total detections: {total_dets}")
    print(f"  Frames with detections: {len(predictions.get('detections', []))}")
    
    # Save output info
    info = {
        'video': str(video_path),
        'model': str(model_path),
        'total_frames': total_frames,
        'fps': fps,
        'duration': total_frames / fps,
        'inference_time': elapsed_time.total_seconds(),
        'fps_inference': total_frames / elapsed_time.total_seconds(),
        'total_detections': total_dets,
        'frames_with_detections': len(predictions.get('detections', [])),
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    info_file = output_dir / 'test_info.json'
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Test info saved to: {info_file}")
    
    # Output files
    print(f"\n📁 Output files:")
    for item in output_dir.glob('**/*'):
        if item.is_file():
            size_mb = item.stat().st_size / (1024*1024)
            print(f"  - {item.relative_to(output_dir)} ({size_mb:.2f} MB)")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_single_video()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
