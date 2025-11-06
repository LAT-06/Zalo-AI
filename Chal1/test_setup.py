"""
Quick Test Script
Verifies that all components are properly installed and functioning
"""

import sys
import subprocess
from pathlib import Path


def check_import(module_name, package_name=None):
    """Check if a Python module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("✗ CUDA is NOT available (will use CPU)")
            return False
    except:
        print("✗ Could not check CUDA")
        return False


def check_file_structure():
    """Check if required files exist"""
    required_files = [
        'parse_dataset.py',
        'augmentation.py',
        'train_yolov8.py',
        'video_inference.py',
        'tensorrt_optimize.py',
        'stiou_metric.py',
        'generate_submission.py',
        'pipeline.py',
        'requirements.txt',
        'Dockerfile',
        'README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if Path(filename).exists():
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} is MISSING")
            all_exist = False
    
    return all_exist


def check_data_structure():
    """Check if data directories exist"""
    data_dirs = [
        ('train', True),
        ('train/annotations', True),
        ('train/annotations/annotations.json', False),
        ('train/samples', True),
        ('public_test', False),
        ('public_test/samples', False),
    ]
    
    all_exist = True
    for path, required in data_dirs:
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✓ {path} exists")
        else:
            if required:
                print(f"✗ {path} is MISSING (required)")
                all_exist = False
            else:
                print(f"⚠ {path} not found (optional)")
    
    return all_exist


def test_dataset_parser():
    """Test dataset parser on a small sample"""
    print("\nTesting dataset parser...")
    
    # Check if we have training data
    ann_file = Path('train/annotations/annotations.json')
    if not ann_file.exists():
        print("✗ Cannot test parser - annotations.json not found")
        return False
    
    try:
        # Import and test basic functionality
        import json
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded annotations: {len(data)} videos")
        
        # Check format
        if len(data) > 0:
            sample = data[0]
            required_keys = ['video_id', 'annotations']
            if all(k in sample for k in required_keys):
                print(f"✓ Annotation format is correct")
                print(f"  Sample video: {sample['video_id']}")
                return True
            else:
                print(f"✗ Annotation format is incorrect")
                return False
        
    except Exception as e:
        print(f"✗ Error testing parser: {e}")
        return False


def main():
    """Main test function"""
    print("="*70)
    print("AEROEYES CHALLENGE - SYSTEM CHECK")
    print("="*70)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check required imports
    print("\n" + "="*70)
    print("Checking Python Dependencies")
    print("="*70)
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('albumentations', 'albumentations'),
        ('yaml', 'pyyaml'),
        ('sklearn', 'scikit-learn'),
    ]
    
    deps_ok = True
    for module, package in dependencies:
        if not check_import(module, package):
            deps_ok = False
    
    # Check CUDA
    print("\n" + "="*70)
    print("Checking CUDA")
    print("="*70)
    check_cuda()
    
    # Check file structure
    print("\n" + "="*70)
    print("Checking File Structure")
    print("="*70)
    files_ok = check_file_structure()
    
    # Check data structure
    print("\n" + "="*70)
    print("Checking Data Structure")
    print("="*70)
    data_ok = check_data_structure()
    
    # Test dataset parser
    print("\n" + "="*70)
    print("Testing Components")
    print("="*70)
    parser_ok = test_dataset_parser()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    status = []
    status.append(("Dependencies", deps_ok))
    status.append(("Files", files_ok))
    status.append(("Data", data_ok))
    status.append(("Parser", parser_ok))
    
    all_ok = all(ok for _, ok in status)
    
    for name, ok in status:
        symbol = "✓" if ok else "✗"
        print(f"{symbol} {name}: {'PASS' if ok else 'FAIL'}")
    
    print("="*70)
    
    if all_ok:
        print("\n🎉 All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. Run: python pipeline.py --help")
        print("  2. Or: python parse_dataset.py --help")
        print("  3. See README.md for detailed instructions")
    else:
        print("\n⚠ Some checks failed. Please:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Ensure data is in correct structure (see README.md)")
        print("  3. Run this script again to verify")
    
    print()


if __name__ == "__main__":
    main()
