"""
Submission File Generator
Format inference results into required JSON format for submission
"""

import json
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict


class SubmissionGenerator:
    """
    Generate submission file from inference results
    Format: {video_id: str, detections: [{frame: int, bboxes: [[x1,y1,x2,y2], ...]}, ...]}
    """
    
    def __init__(self):
        self.submissions = []
    
    def load_inference_results(self, results_dir):
        """
        Load all inference results from a directory
        
        Args:
            results_dir: Directory containing inference JSON files
        
        Returns:
            List of video results
        """
        results_dir = Path(results_dir)
        results = []
        
        # Find all JSON files
        json_files = list(results_dir.glob("*.json"))
        
        print(f"Found {len(json_files)} result files")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        
        return results
    
    def validate_format(self, submission):
        """
        Validate submission format
        
        Args:
            submission: Submission dictionary
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['video_id', 'detections']
        
        # Check required keys
        for key in required_keys:
            if key not in submission:
                print(f"ERROR: Missing required key '{key}'")
                return False
        
        # Check video_id is string
        if not isinstance(submission['video_id'], str):
            print(f"ERROR: video_id must be string, got {type(submission['video_id'])}")
            return False
        
        # Check detections is list
        if not isinstance(submission['detections'], list):
            print(f"ERROR: detections must be list, got {type(submission['detections'])}")
            return False
        
        # Check each detection
        for det in submission['detections']:
            if 'frame' not in det or 'bboxes' not in det:
                print(f"ERROR: Detection missing 'frame' or 'bboxes'")
                return False
            
            if not isinstance(det['frame'], int):
                print(f"ERROR: frame must be int, got {type(det['frame'])}")
                return False
            
            if not isinstance(det['bboxes'], list):
                print(f"ERROR: bboxes must be list, got {type(det['bboxes'])}")
                return False
            
            # Check bbox format
            for bbox in det['bboxes']:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print(f"ERROR: bbox must be list of 4 numbers, got {bbox}")
                    return False
                
                # Check bbox values are numbers
                if not all(isinstance(x, (int, float)) for x in bbox):
                    print(f"ERROR: bbox values must be numbers, got {bbox}")
                    return False
                
                # Check bbox is in correct format [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    print(f"WARNING: Invalid bbox format {bbox} (x2 <= x1 or y2 <= y1)")
        
        return True
    
    def add_empty_video(self, video_id):
        """
        Add a video with no detections
        
        Args:
            video_id: Video identifier
        """
        submission = {
            "video_id": video_id,
            "detections": []
        }
        
        self.submissions.append(submission)
    
    def add_inference_result(self, result):
        """
        Add inference result to submissions
        
        Args:
            result: Inference result dictionary
        """
        # Validate format
        if not self.validate_format(result):
            raise ValueError(f"Invalid format for video {result.get('video_id', 'unknown')}")
        
        self.submissions.append(result)
    
    def ensure_all_videos(self, video_list):
        """
        Ensure all videos in list are present in submission
        
        Args:
            video_list: List of all video IDs that must appear
        """
        existing_videos = {sub['video_id'] for sub in self.submissions}
        
        for video_id in video_list:
            if video_id not in existing_videos:
                print(f"Adding empty submission for video: {video_id}")
                self.add_empty_video(video_id)
    
    def generate_submission(self, output_path, timestamp=True):
        """
        Generate submission file
        
        Args:
            output_path: Path to save submission JSON
            timestamp: Add timestamp to filename
        """
        output_path = Path(output_path)
        
        # Add timestamp to filename
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path.parent / f"submission_{timestamp_str}.json"
        
        # Sort submissions by video_id for consistency
        self.submissions.sort(key=lambda x: x['video_id'])
        
        # Validate all submissions
        print("Validating submission format...")
        for sub in self.submissions:
            if not self.validate_format(sub):
                raise ValueError(f"Validation failed for {sub['video_id']}")
        
        # Save to file
        print(f"Saving submission to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(self.submissions, f, indent=2)
        
        # Print statistics
        total_videos = len(self.submissions)
        videos_with_detections = sum(1 for sub in self.submissions 
                                     if len(sub['detections']) > 0)
        videos_without_detections = total_videos - videos_with_detections
        
        total_frames = sum(len(sub['detections']) for sub in self.submissions)
        total_bboxes = sum(
            sum(len(det['bboxes']) for det in sub['detections'])
            for sub in self.submissions
        )
        
        print("\n" + "="*60)
        print("Submission Statistics")
        print("="*60)
        print(f"Total videos:              {total_videos}")
        print(f"Videos with detections:    {videos_with_detections}")
        print(f"Videos without detections: {videos_without_detections}")
        print(f"Total frames with objects: {total_frames}")
        print(f"Total bounding boxes:      {total_bboxes}")
        print("="*60 + "\n")
        
        print(f"✓ Submission file generated: {output_path}")
        
        return output_path


def get_test_video_list(test_dir):
    """
    Get list of all test videos
    
    Args:
        test_dir: Directory containing test samples
    
    Returns:
        List of video IDs
    """
    test_dir = Path(test_dir)
    video_dirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
    return sorted(video_dirs)


def main():
    """Main submission generation function"""
    parser = argparse.ArgumentParser(
        description='Generate submission file from inference results'
    )
    
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing inference JSON files')
    parser.add_argument('--output', type=str, default='submission.json',
                        help='Output submission file path')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Test directory to ensure all videos are included')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='Do not add timestamp to filename')
    
    # Validation
    parser.add_argument('--val_gt', type=str, default=None,
                        help='Validation ground truth for sanity check')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SubmissionGenerator()
    
    # Load inference results
    print("Loading inference results...")
    results = generator.load_inference_results(args.results_dir)
    
    print(f"Loaded {len(results)} inference results")
    
    # Add results to submission
    for result in results:
        generator.add_inference_result(result)
    
    # Ensure all test videos are present
    if args.test_dir:
        print("Checking for missing test videos...")
        test_videos = get_test_video_list(args.test_dir)
        generator.ensure_all_videos(test_videos)
    
    # Generate submission file
    submission_path = generator.generate_submission(
        args.output,
        timestamp=not args.no_timestamp
    )
    
    # Sanity check on validation set
    if args.val_gt:
        print("\nRunning sanity check on validation set...")
        try:
            from stiou_metric import STIoUMetric
            
            metric = STIoUMetric()
            mean_stiou, per_video_scores = metric.evaluate(
                str(submission_path),
                args.val_gt
            )
            
            print(f"\nEstimated ST-IoU on validation set: {mean_stiou:.4f}")
            
            # Show top and bottom performing videos
            sorted_scores = sorted(per_video_scores.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            print("\nTop 5 videos:")
            for video_id, score in sorted_scores[:5]:
                print(f"  {video_id}: {score:.4f}")
            
            print("\nBottom 5 videos:")
            for video_id, score in sorted_scores[-5:]:
                print(f"  {video_id}: {score:.4f}")
        
        except ImportError:
            print("WARNING: Could not import stiou_metric for validation")
        except Exception as e:
            print(f"WARNING: Sanity check failed: {e}")


if __name__ == "__main__":
    main()
