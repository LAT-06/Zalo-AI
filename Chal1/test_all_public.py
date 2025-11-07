"""
Quick Test on All Public Test Videos
Run inference on all 6 videos in public_test/samples
"""

import os
import json
import cv2
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from video_inference import VideoInference
import torch


def test_all_videos():
    """Test model on all public_test videos"""
    
    print("\n" + "="*70)
    print("TESTING ON ALL PUBLIC_TEST VIDEOS")
    print("="*70 + "\n")
    
    # Model path
    model_path = Path('runs/train/aeroeyes/weights/best.pt')
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"✓ Model: {model_path}\n")
    
    # Get all videos
    video_dir = Path('public_test/samples')
    videos = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(videos)} videos:\n")
    
    results = []
    total_detections = 0
    total_time = 0
    
    for i, video_folder in enumerate(videos, 1):
        video_name = video_folder.name
        video_file = video_folder / 'drone_video.mp4'
        
        if not video_file.exists():
            print(f"❌ {i}. {video_name}: Video not found")
            continue
        
        print(f"⏳ {i}. {video_name}...", end=" ", flush=True)
        
        # Get video info
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        # Output directory
        output_dir = Path('test_output') / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        start_time = datetime.now()
        
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
            video_path=str(video_file),
            output_json=str(output_dir / 'predictions.json'),
            visualize=False,
            output_video=None  # Skip video output for speed
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        total_time += elapsed
        
        # Count detections
        num_detections = sum(len(d.get('bboxes', [])) for d in predictions.get('detections', []))
        total_detections += num_detections
        
        fps_inference = total_frames / elapsed
        
        print(f"✓ {num_detections:3d} detections | {fps_inference:.1f} FPS | {elapsed:.1f}s")
        
        results.append({
            'video': video_name,
            'total_frames': total_frames,
            'duration': duration,
            'detections': num_detections,
            'inference_time': elapsed,
            'fps_inference': fps_inference
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Video Name':<20} {'Frames':>8} {'Duration':>10} {'Detections':>12} {'Inf. Time':>10} {'FPS':>8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['video']:<20} {result['total_frames']:>8} "
              f"{result['duration']:>10.1f}s {result['detections']:>12} "
              f"{result['inference_time']:>10.1f}s {result['fps_inference']:>8.1f}")
    
    print("-" * 70)
    total_frames = sum(r['total_frames'] for r in results)
    print(f"{'TOTAL':<20} {total_frames:>8} {total_frames/25/60:>10.1f}m "
          f"{total_detections:>12} {total_time:>10.1f}s {total_frames/total_time:>8.1f}")
    
    print("\n✓ Test complete!")
    print(f"\nResults saved to test_output/*/predictions.json")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': str(model_path),
        'total_videos': len(results),
        'total_detections': total_detections,
        'total_inference_time': total_time,
        'average_fps': total_frames / total_time,
        'videos': results
    }
    
    summary_file = Path('test_output/summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        test_all_videos()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
