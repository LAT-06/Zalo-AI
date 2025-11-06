"""
ST-IoU (Spatio-Temporal Intersection-over-Union) Metric Implementation
Evaluates spatio-temporal detection quality
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import argparse


class STIoUMetric:
    """
    Spatio-Temporal IoU metric for video object detection
    
    Formula: ST-IoU = Σ(IoU(Bf, Bf')) / |frames_union|
    where:
    - intersection: frames with both prediction and ground-truth
    - union: all frames with either prediction or ground-truth
    - IoU(Bf, Bf'): spatial IoU at frame f
    """
    
    def __init__(self, iou_threshold=0.5):
        """
        Initialize ST-IoU metric
        
        Args:
            iou_threshold: IoU threshold for considering a detection as correct
        """
        self.iou_threshold = iou_threshold
    
    def bbox_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        
        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        # IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def match_bboxes(self, pred_bboxes, gt_bboxes):
        """
        Match predicted bboxes to ground truth using Hungarian algorithm
        
        Args:
            pred_bboxes: List of predicted bboxes
            gt_bboxes: List of ground truth bboxes
        
        Returns:
            List of matched IoU values
        """
        if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            return []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
        for i, pred_box in enumerate(pred_bboxes):
            for j, gt_box in enumerate(gt_bboxes):
                iou_matrix[i, j] = self.bbox_iou(pred_box, gt_box)
        
        # Greedy matching (can be replaced with Hungarian algorithm)
        matched_ious = []
        used_preds = set()
        used_gts = set()
        
        # Sort by IoU in descending order
        matches = []
        for i in range(len(pred_bboxes)):
            for j in range(len(gt_bboxes)):
                matches.append((iou_matrix[i, j], i, j))
        matches.sort(reverse=True)
        
        for iou, i, j in matches:
            if i not in used_preds and j not in used_gts:
                matched_ious.append(iou)
                used_preds.add(i)
                used_gts.add(j)
        
        return matched_ious
    
    def calculate_video_stiou(self, predictions, ground_truth):
        """
        Calculate ST-IoU for a single video
        
        Args:
            predictions: Dict with frame -> list of bboxes
            ground_truth: Dict with frame -> list of bboxes
        
        Returns:
            ST-IoU score for the video
        """
        # Get union of frames
        pred_frames = set(predictions.keys())
        gt_frames = set(ground_truth.keys())
        union_frames = pred_frames | gt_frames
        intersection_frames = pred_frames & gt_frames
        
        if len(union_frames) == 0:
            return 0.0
        
        # Calculate IoU sum over intersection frames
        iou_sum = 0.0
        
        for frame in intersection_frames:
            pred_bboxes = predictions[frame]
            gt_bboxes = ground_truth[frame]
            
            # Match bboxes and get IoUs
            matched_ious = self.match_bboxes(pred_bboxes, gt_bboxes)
            
            # Sum IoUs for this frame
            iou_sum += sum(matched_ious)
        
        # ST-IoU = sum of IoUs / number of union frames
        stiou = iou_sum / len(union_frames)
        
        return stiou
    
    def calculate_dataset_stiou(self, predictions_dict, ground_truth_dict):
        """
        Calculate ST-IoU across entire dataset
        
        Args:
            predictions_dict: Dict of video_id -> {frame -> bboxes}
            ground_truth_dict: Dict of video_id -> {frame -> bboxes}
        
        Returns:
            Mean ST-IoU and per-video scores
        """
        per_video_scores = {}
        
        # Calculate ST-IoU for each video
        all_videos = set(predictions_dict.keys()) | set(ground_truth_dict.keys())
        
        for video_id in all_videos:
            pred = predictions_dict.get(video_id, {})
            gt = ground_truth_dict.get(video_id, {})
            
            stiou = self.calculate_video_stiou(pred, gt)
            per_video_scores[video_id] = stiou
        
        # Calculate mean ST-IoU
        mean_stiou = np.mean(list(per_video_scores.values())) if per_video_scores else 0.0
        
        return mean_stiou, per_video_scores
    
    def load_predictions(self, pred_json):
        """
        Load predictions from JSON file
        
        Args:
            pred_json: Path to predictions JSON
        
        Returns:
            Dict of video_id -> {frame -> list of bboxes}
        """
        with open(pred_json, 'r') as f:
            data = json.load(f)
        
        predictions_dict = {}
        
        # Handle both single video and multiple videos format
        if isinstance(data, dict) and 'video_id' in data:
            # Single video format
            data = [data]
        
        for video_data in data:
            video_id = video_data['video_id']
            frame_dict = {}
            
            for det in video_data.get('detections', []):
                frame = det['frame']
                bboxes = det.get('bboxes', [])
                if len(bboxes) > 0:
                    frame_dict[frame] = bboxes
            
            predictions_dict[video_id] = frame_dict
        
        return predictions_dict
    
    def load_ground_truth(self, gt_json):
        """
        Load ground truth from annotations JSON
        
        Args:
            gt_json: Path to ground truth annotations
        
        Returns:
            Dict of video_id -> {frame -> list of bboxes}
        """
        with open(gt_json, 'r') as f:
            data = json.load(f)
        
        gt_dict = {}
        
        for video_data in data:
            video_id = video_data['video_id']
            frame_dict = {}
            
            for ann_obj in video_data.get('annotations', []):
                for bbox_data in ann_obj.get('bboxes', []):
                    frame = bbox_data['frame']
                    bbox = [bbox_data['x1'], bbox_data['y1'], 
                           bbox_data['x2'], bbox_data['y2']]
                    
                    if frame not in frame_dict:
                        frame_dict[frame] = []
                    frame_dict[frame].append(bbox)
            
            gt_dict[video_id] = frame_dict
        
        return gt_dict
    
    def evaluate(self, pred_json, gt_json):
        """
        Evaluate predictions against ground truth
        
        Args:
            pred_json: Path to predictions JSON
            gt_json: Path to ground truth JSON
        
        Returns:
            Mean ST-IoU and per-video scores
        """
        print("Loading predictions...")
        predictions_dict = self.load_predictions(pred_json)
        
        print("Loading ground truth...")
        gt_dict = self.load_ground_truth(gt_json)
        
        print(f"Evaluating {len(predictions_dict)} predictions against {len(gt_dict)} ground truths...")
        
        mean_stiou, per_video_scores = self.calculate_dataset_stiou(
            predictions_dict, gt_dict
        )
        
        return mean_stiou, per_video_scores


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Calculate ST-IoU metric'
    )
    
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth JSON file')
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='IoU threshold')
    
    args = parser.parse_args()
    
    # Calculate ST-IoU
    metric = STIoUMetric(iou_threshold=args.iou_thresh)
    mean_stiou, per_video_scores = metric.evaluate(args.pred, args.gt)
    
    # Print results
    print("\n" + "="*60)
    print("ST-IoU Evaluation Results")
    print("="*60)
    print(f"Mean ST-IoU: {mean_stiou:.4f}")
    print("\nPer-video scores:")
    print("-"*60)
    
    # Sort by score
    sorted_scores = sorted(per_video_scores.items(), key=lambda x: x[1], reverse=True)
    
    for video_id, score in sorted_scores[:20]:  # Show top 20
        print(f"{video_id:30s}: {score:.4f}")
    
    if len(sorted_scores) > 20:
        print(f"... and {len(sorted_scores) - 20} more videos")
    
    print("="*60 + "\n")
    
    # Save detailed results
    output_file = Path(args.pred).parent / "stiou_results.json"
    results = {
        'mean_stiou': float(mean_stiou),
        'per_video_scores': {k: float(v) for k, v in per_video_scores.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
