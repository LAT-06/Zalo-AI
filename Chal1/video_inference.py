"""
Video Inference Script with ByteTrack Integration
Process drone videos frame-by-frame with object tracking
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict, deque
from ultralytics import YOLO
import torch


class ByteTracker:
    """
    Simplified ByteTrack implementation for object tracking
    Maintains object IDs across frames with occlusion handling
    """
    
    def __init__(self, track_thresh=0.4, track_buffer=30, match_thresh=0.8):
        """
        Initialize ByteTracker
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching detections to tracks
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_objects = {}  # track_id -> track_info
        self.next_track_id = 1
        self.frame_id = 0
        
    def iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of (bbox, confidence, class_id) tuples
                       bbox in format [x1, y1, x2, y2]
        
        Returns:
            List of (bbox, track_id, class_id) tuples
        """
        self.frame_id += 1
        
        # Filter detections by threshold
        valid_detections = [d for d in detections if d[1] >= self.track_thresh]
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = []
        
        if len(self.tracked_objects) > 0 and len(valid_detections) > 0:
            # Calculate IoU matrix
            track_ids = list(self.tracked_objects.keys())
            iou_matrix = np.zeros((len(valid_detections), len(track_ids)))
            
            for i, det in enumerate(valid_detections):
                det_box = det[0]
                for j, track_id in enumerate(track_ids):
                    track_box = self.tracked_objects[track_id]['bbox']
                    iou_matrix[i, j] = self.iou(det_box, track_box)
            
            # Greedy matching
            matched_indices = []
            for _ in range(min(len(valid_detections), len(track_ids))):
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[i, j] >= self.match_thresh:
                    track_id = track_ids[j]
                    det_box, det_conf, det_class = valid_detections[i]
                    
                    # Update track
                    self.tracked_objects[track_id]['bbox'] = det_box
                    self.tracked_objects[track_id]['confidence'] = det_conf
                    self.tracked_objects[track_id]['class_id'] = det_class
                    self.tracked_objects[track_id]['last_frame'] = self.frame_id
                    self.tracked_objects[track_id]['lost_frames'] = 0
                    
                    matched_tracks.append((det_box, track_id, det_class))
                    matched_indices.append(i)
                    
                    # Zero out matched entries
                    iou_matrix[i, :] = -1
                    iou_matrix[:, j] = -1
                else:
                    break
            
            # Unmatched detections
            unmatched_detections = [d for i, d in enumerate(valid_detections) 
                                   if i not in matched_indices]
        else:
            unmatched_detections = valid_detections
        
        # Create new tracks for unmatched detections
        for det_box, det_conf, det_class in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracked_objects[track_id] = {
                'bbox': det_box,
                'confidence': det_conf,
                'class_id': det_class,
                'last_frame': self.frame_id,
                'lost_frames': 0
            }
            
            matched_tracks.append((det_box, track_id, det_class))
        
        # Update lost tracks
        lost_track_ids = []
        for track_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[track_id]['last_frame'] < self.frame_id:
                self.tracked_objects[track_id]['lost_frames'] += 1
                
                # Remove tracks that have been lost for too long
                if self.tracked_objects[track_id]['lost_frames'] > self.track_buffer:
                    lost_track_ids.append(track_id)
        
        for track_id in lost_track_ids:
            del self.tracked_objects[track_id]
        
        return matched_tracks
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_objects = {}
        self.next_track_id = 1
        self.frame_id = 0


class TemporalSmoother:
    """
    Temporal smoothing using moving average for bbox coordinates
    Reduces jitter in tracking results
    """
    
    def __init__(self, window_size=5):
        """
        Initialize smoother
        
        Args:
            window_size: Number of frames for moving average
        """
        self.window_size = window_size
        self.history = defaultdict(lambda: deque(maxlen=window_size))
    
    def smooth(self, track_id, bbox):
        """
        Apply temporal smoothing to bbox
        
        Args:
            track_id: Object track ID
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Smoothed bbox
        """
        self.history[track_id].append(bbox)
        
        # Calculate moving average
        history_array = np.array(list(self.history[track_id]))
        smoothed_bbox = np.mean(history_array, axis=0)
        
        return smoothed_bbox.astype(int).tolist()
    
    def reset(self):
        """Reset smoother history"""
        self.history.clear()


class VideoInference:
    """
    Video inference pipeline with tracking
    """
    
    def __init__(
        self,
        model_path,
        conf_thresh=0.25,
        iou_thresh=0.45,
        track_thresh=0.4,
        track_buffer=30,
        match_thresh=0.8,
        smooth_window=5,
        device='cuda'
    ):
        """
        Initialize video inference pipeline
        
        Args:
            model_path: Path to YOLOv8 model weights
            conf_thresh: Confidence threshold for detections
            iou_thresh: NMS IoU threshold
            track_thresh: Tracking confidence threshold
            track_buffer: Frame buffer for lost tracks
            match_thresh: IoU threshold for track matching
            smooth_window: Temporal smoothing window size
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        self.tracker = ByteTracker(track_thresh, track_buffer, match_thresh)
        self.smoother = TemporalSmoother(smooth_window)
    
    def process_video(self, video_path, output_json=None, visualize=False, output_video=None):
        """
        Process video and generate predictions
        
        Args:
            video_path: Path to input video
            output_json: Path to save JSON predictions (optional)
            visualize: Whether to visualize results
            output_video: Path to save visualization video (optional)
        
        Returns:
            Dictionary with video_id and detections
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        
        print(f"Processing video: {video_id}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        
        # Setup video writer if visualizing
        writer = None
        if visualize and output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Reset tracker
        self.tracker.reset()
        self.smoother.reset()
        
        # Process frames
        frame_detections = {}
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, 
                                verbose=False)[0]
            
            # Extract detections
            detections = []
            if len(results.boxes) > 0:
                for box in results.boxes:
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append((bbox.tolist(), conf, cls_id))
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Apply temporal smoothing and collect frame detections
            frame_bboxes = []
            for bbox, track_id, cls_id in tracked_objects:
                smoothed_bbox = self.smoother.smooth(track_id, bbox)
                frame_bboxes.append(smoothed_bbox)
                
                # Visualize if requested
                if visualize:
                    x1, y1, x2, y2 = [int(v) for v in smoothed_bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{track_id} Cls:{cls_id}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store detections for this frame (only if non-empty)
            if len(frame_bboxes) > 0:
                frame_detections[frame_idx] = frame_bboxes
            
            # Write frame if visualizing
            if writer:
                writer.write(frame)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"Completed processing {frame_idx} frames")
        print(f"Frames with detections: {len(frame_detections)}")
        
        # Format output
        output_data = {
            "video_id": video_id,
            "detections": [
                {"frame": frame, "bboxes": bboxes}
                for frame, bboxes in sorted(frame_detections.items())
            ]
        }
        
        # Save to JSON if requested
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved predictions to: {output_json}")
        
        return output_data


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Video inference with tracking')
    
    # Input/Output
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLOv8 model weights')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output JSON file')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize tracking results')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Path to output visualization video')
    
    # Detection parameters
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    
    # Tracking parameters
    parser.add_argument('--track_thresh', type=float, default=0.4,
                        help='Tracking confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=30,
                        help='Frame buffer for lost tracks')
    parser.add_argument('--match_thresh', type=float, default=0.8,
                        help='IoU threshold for track matching')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='Temporal smoothing window size')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Initialize inference pipeline
    pipeline = VideoInference(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        smooth_window=args.smooth_window,
        device=args.device
    )
    
    # Process video
    pipeline.process_video(
        video_path=args.video,
        output_json=args.output,
        visualize=args.visualize,
        output_video=args.output_video
    )


if __name__ == "__main__":
    main()
