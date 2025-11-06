"""
Data Augmentation Pipeline for Aerial Drone Footage
Uses Albumentations library with YOLO format support
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import random


class AerialAugmentation:
    """
    Augmentation pipeline specifically designed for aerial drone footage
    Handles small objects carefully and applies drone-specific transforms
    """
    
    def __init__(self, img_size=640, mosaic_prob=0.5, mixup_prob=0.2):
        """
        Initialize augmentation pipeline
        
        Args:
            img_size: Target image size (default 640 for YOLOv8)
            mosaic_prob: Probability of applying mosaic augmentation
            mixup_prob: Probability of applying mixup augmentation
        """
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        
        # Training augmentation pipeline
        self.train_transform = A.Compose([
            # Random scale (0.5-1.5x)
            A.RandomScale(scale_limit=0.5, p=0.5),
            
            # Random rotation (-10 to +10 degrees)
            # Small angles to preserve aerial perspective
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            
            # HSV color jitter
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),  # Convert to degrees
                sat_shift_limit=int(0.7 * 100),
                val_shift_limit=int(0.4 * 100),
                p=0.7
            ),
            
            # Random brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Horizontal flip only (not vertical for aerial views)
            A.HorizontalFlip(p=0.5),
            
            # Motion blur to simulate drone movement
            A.MotionBlur(blur_limit=7, p=0.3),
            
            # Additional blur for realism
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.2),
            
            # Random fog/haze (common in aerial footage)
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            
            # Random shadow
            A.RandomShadow(p=0.1),
            
            # Pad to square and resize
            A.LongestMaxSize(max_size=self.img_size, p=1.0),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,  # Gray padding
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_area=16,  # Minimum area to keep small objects
            min_visibility=0.3,
            label_fields=['class_labels']
        ))
        
        # Validation transform (no augmentation, only resize)
        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=self.img_size, p=1.0),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=114,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def mosaic_augmentation(self, images, bboxes_list, class_labels_list):
        """
        Mosaic augmentation - combine 4 images into one
        
        Args:
            images: List of 4 images
            bboxes_list: List of 4 bbox arrays in YOLO format
            class_labels_list: List of 4 class label arrays
            
        Returns:
            Mosaic image and adjusted bboxes
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        
        mosaic_h = mosaic_w = self.img_size
        
        # Random center point
        yc = int(random.uniform(mosaic_h * 0.4, mosaic_h * 0.6))
        xc = int(random.uniform(mosaic_w * 0.4, mosaic_w * 0.6))
        
        # Create mosaic canvas
        mosaic_img = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, class_labels_list)):
            h, w = img.shape[:2]
            
            # Resize image to fit in quadrant
            scale = min(mosaic_h / h, mosaic_w / w) * 0.5
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Place image in correct quadrant
            if i == 0:  # Top-left
                x1a, y1a = max(xc - new_w, 0), max(yc - new_h, 0)
                x2a, y2a = xc, yc
                x1b, y1b = new_w - (x2a - x1a), new_h - (y2a - y1a)
                x2b, y2b = new_w, new_h
            elif i == 1:  # Top-right
                x1a, y1a = xc, max(yc - new_h, 0)
                x2a, y2a = min(xc + new_w, mosaic_w), yc
                x1b, y1b = 0, new_h - (y2a - y1a)
                x2b, y2b = min(new_w, x2a - x1a), new_h
            elif i == 2:  # Bottom-left
                x1a, y1a = max(xc - new_w, 0), yc
                x2a, y2a = xc, min(yc + new_h, mosaic_h)
                x1b, y1b = new_w - (x2a - x1a), 0
                x2b, y2b = new_w, min(new_h, y2a - y1a)
            else:  # Bottom-right
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + new_w, mosaic_w), min(yc + new_h, mosaic_h)
                x1b, y1b = 0, 0
                x2b, y2b = min(new_w, x2a - x1a), min(new_h, y2a - y1a)
            
            # Place image on canvas
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized[y1b:y2b, x1b:x2b]
            
            # Adjust bboxes (convert YOLO to absolute, translate, convert back)
            if len(bboxes) > 0:
                for bbox, label in zip(bboxes, labels):
                    x_center, y_center, bbox_w, bbox_h = bbox
                    
                    # YOLO to absolute in resized image
                    x_abs = x_center * new_w
                    y_abs = y_center * new_h
                    w_abs = bbox_w * new_w
                    h_abs = bbox_h * new_h
                    
                    # Translate to mosaic coordinates
                    x_abs += (x1a - x1b)
                    y_abs += (y1a - y1b)
                    
                    # Convert back to YOLO format (normalized by mosaic size)
                    x_center_new = x_abs / mosaic_w
                    y_center_new = y_abs / mosaic_h
                    w_new = w_abs / mosaic_w
                    h_new = h_abs / mosaic_h
                    
                    # Only keep bboxes that are mostly inside the image
                    if 0 < x_center_new < 1 and 0 < y_center_new < 1:
                        # Clip to image boundaries
                        x_center_new = np.clip(x_center_new, w_new/2, 1 - w_new/2)
                        y_center_new = np.clip(y_center_new, h_new/2, 1 - h_new/2)
                        
                        mosaic_bboxes.append([x_center_new, y_center_new, w_new, h_new])
                        mosaic_labels.append(label)
        
        return mosaic_img, np.array(mosaic_bboxes), np.array(mosaic_labels)
    
    def mixup_augmentation(self, img1, bboxes1, labels1, img2, bboxes2, labels2, alpha=0.2):
        """
        MixUp augmentation - blend two images
        
        Args:
            img1, img2: Input images
            bboxes1, bboxes2: Bounding boxes in YOLO format
            labels1, labels2: Class labels
            alpha: MixUp alpha parameter
            
        Returns:
            Mixed image and combined bboxes
        """
        # Random mixing ratio
        lam = np.random.beta(alpha, alpha)
        
        # Ensure images are same size
        h, w = img1.shape[:2]
        img2_resized = cv2.resize(img2, (w, h))
        
        # Mix images
        mixed_img = (lam * img1 + (1 - lam) * img2_resized).astype(np.uint8)
        
        # Combine bboxes (keep all boxes from both images)
        mixed_bboxes = np.vstack([bboxes1, bboxes2]) if len(bboxes1) > 0 and len(bboxes2) > 0 else \
                       bboxes1 if len(bboxes1) > 0 else bboxes2
        mixed_labels = np.concatenate([labels1, labels2]) if len(labels1) > 0 and len(labels2) > 0 else \
                       labels1 if len(labels1) > 0 else labels2
        
        return mixed_img, mixed_bboxes, mixed_labels
    
    def __call__(self, image, bboxes, class_labels, is_train=True, extra_images=None, 
                 extra_bboxes=None, extra_labels=None):
        """
        Apply augmentation pipeline
        
        Args:
            image: Input image
            bboxes: Bounding boxes in YOLO format (x_center, y_center, width, height)
            class_labels: Class labels for each bbox
            is_train: Whether to apply training augmentations
            extra_images: Additional images for mosaic/mixup
            extra_bboxes: Additional bboxes for mosaic/mixup
            extra_labels: Additional labels for mosaic/mixup
            
        Returns:
            Augmented image, bboxes, and labels
        """
        # Apply mosaic augmentation
        if is_train and extra_images is not None and len(extra_images) >= 3 and \
           random.random() < self.mosaic_prob:
            # Use first 3 extra images plus current image
            images = [image] + extra_images[:3]
            bboxes_list = [bboxes] + extra_bboxes[:3]
            labels_list = [class_labels] + extra_labels[:3]
            
            image, bboxes, class_labels = self.mosaic_augmentation(
                images, bboxes_list, labels_list
            )
        
        # Apply standard augmentations
        transform = self.train_transform if is_train else self.val_transform
        
        try:
            transformed = transform(
                image=image,
                bboxes=bboxes if len(bboxes) > 0 else [],
                class_labels=class_labels if len(class_labels) > 0 else []
            )
            
            image = transformed['image']
            bboxes = np.array(transformed['bboxes'])
            class_labels = np.array(transformed['class_labels'])
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            # Return original
            pass
        
        # Apply mixup augmentation
        if is_train and extra_images is not None and len(extra_images) >= 1 and \
           random.random() < self.mixup_prob:
            # Use first extra image for mixup
            image, bboxes, class_labels = self.mixup_augmentation(
                image, bboxes, class_labels,
                extra_images[0], extra_bboxes[0], extra_labels[0],
                alpha=0.2
            )
        
        return image, bboxes, class_labels


def test_augmentation():
    """Test augmentation pipeline"""
    # Create dummy data
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    bboxes = np.array([[0.5, 0.5, 0.2, 0.2]])  # YOLO format
    labels = np.array([0])
    
    # Test augmentation
    aug = AerialAugmentation(img_size=640)
    
    # Test normal augmentation
    aug_img, aug_bboxes, aug_labels = aug(img, bboxes, labels, is_train=True)
    print(f"Augmented image shape: {aug_img.shape}")
    print(f"Augmented bboxes: {aug_bboxes}")
    
    # Test mosaic augmentation
    extra_images = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    extra_bboxes = [np.array([[0.3, 0.3, 0.1, 0.1]]) for _ in range(3)]
    extra_labels = [np.array([0]) for _ in range(3)]
    
    aug_img, aug_bboxes, aug_labels = aug(
        img, bboxes, labels,
        is_train=True,
        extra_images=extra_images,
        extra_bboxes=extra_bboxes,
        extra_labels=extra_labels
    )
    print(f"Mosaic augmented image shape: {aug_img.shape}")
    print(f"Mosaic augmented bboxes count: {len(aug_bboxes)}")


if __name__ == "__main__":
    test_augmentation()
