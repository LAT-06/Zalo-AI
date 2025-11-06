### 1.1 Dataset Characteristics
- **Input**: Drone aerial videos (3-5 minutes, 25 fps, 1280x720 resolution)
- **Objects**: Small objects from aerial perspective (backpack, laptop, person, etc.)
- **Challenges**: 
  - Object occlusion and re-appearance
  - Variable lighting conditions
  - Motion blur from drone movement
  - Small object sizes (often < 50x50 pixels)

### 1.2 Task Requirements
- Detect and track objects across video frames
- Maintain consistent object IDs through occlusions
- Output spatio-temporal bounding box predictions
- Evaluate using ST-IoU (Spatio-Temporal Intersection over Union)

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Pipeline Overview

```
Input: Drone Videos + Annotations
    |
    v
[STEP 1] Dataset Parsing & Preprocessing
    |
    v
[STEP 2] Data Augmentation Pipeline
    |
    v
[STEP 3] YOLOv8 Training
    |
    v
[STEP 4] Video Inference + ByteTrack
    |
    v
[STEP 5] ST-IoU Evaluation & Submission
    |
    v
Output: submission.json
```

---

## 3. METHODOLOGY

### 3.1 Dataset Parsing

#### 3.1.1 Annotation Format Conversion

**Input Format (AeroEyes):**
```
{
  "video_id": "Backpack_0",
  "annotations": [{
    "bboxes": [{
      "frame": 3483,
      "x1": 321, "y1": 0,
      "x2": 381, "y2": 12
    }]
  }]
}
```

**Output Format (YOLO):**
```
class_id x_center y_center width height
```

#### 3.1.2 Coordinate Transformation

Given bbox coordinates (x1, y1, x2, y2) and image dimensions (W, H):

**Center Point:**
```
x_center = (x1 + x2) / (2 * W)
y_center = (y1 + y2) / (2 * H)
```

**Dimensions:**
```
width = (x2 - x1) / W
height = (y2 - y1) / H
```

**Normalization:** All values are in range [0, 1]

#### 3.1.3 Frame Sampling Strategy

- Extract all frames with ground truth annotations
- Sample negative frames (no objects) every 50 frames (2 seconds at 25 fps)
- Total frames per video: approximately 200-300 frames
- Reduces computational cost while maintaining detection coverage

#### 3.1.4 Train/Validation Split

- Split: 80% training, 20% validation
- Method: Random split (stratification not used due to small dataset)
- 14 videos total: 11 training, 3 validation

---

### 3.2 Data Augmentation

#### 3.2.1 Mosaic Augmentation

**Algorithm:**
1. Randomly select 4 images
2. Resize each to fit in quadrant
3. Place at random center point (xc, yc)
4. Adjust bounding box coordinates

**Mathematical Formulation:**

For image i in quadrant q, new bbox coordinates:
```
x_new = (x_old * scale_i + offset_x_q) / mosaic_width
y_new = (y_old * scale_i + offset_y_q) / mosaic_height
```

**Benefits:**
- Increases batch diversity
- Improves small object detection
- Simulates multiple objects in scene

#### 3.2.2 Random Scaling

**Scale Range:** s ∈ [0.5, 1.5]

For image I with dimensions (W, H):
```
W_new = W * s
H_new = H * s
```

Corresponding bbox transformation:
```
x_center_new = x_center * s / W_new
y_center_new = y_center * s / H_new
width_new = width * s / W_new
height_new = height * s / H_new
```

#### 3.2.3 Rotation

**Rotation Angle:** θ ∈ [-10°, +10°]

**Why Limited Rotation?**
- Aerial perspective constraint: sky always up, ground always down
- Large rotations (±90°) break semantic meaning
- Small rotations simulate drone tilt

**Rotation Matrix:**
```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

#### 3.2.4 HSV Color Jitter

**Transformation:**
```
H_new = H + δh * 180°     where δh ∈ [-0.015, 0.015]
S_new = S * (1 + δs)      where δs ∈ [-0.7, 0.7]
V_new = V * (1 + δv)      where δv ∈ [-0.4, 0.4]
```

**Purpose:** Simulate varying lighting conditions (dawn, noon, dusk, cloudy)

#### 3.2.5 MixUp Augmentation

**Formula:**
```
x_mixed = λ * x_i + (1 - λ) * x_j
y_mixed = {y_i ∪ y_j}

where λ ~ Beta(α, α), α = 0.2
```

**Effect:** Creates soft samples, improves model generalization

---

### 3.3 YOLOv8 Training

#### 3.3.1 Model Architecture

**Backbone: CSPDarknet**
- Cross Stage Partial connections
- Reduces computational redundancy
- Maintains gradient flow

**Neck: Path Aggregation Network (PAN)**
- Bottom-up and top-down feature fusion
- Multi-scale feature extraction

**Head: Decoupled Detection Head**
- Separate branches for classification and localization
- Anchor-free detection

#### 3.3.2 Loss Function

**Total Loss:**
```
L_total = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl
```

where λ_box = 7.5, λ_cls = 0.5, λ_dfl = 1.5

**1. Box Loss (CIoU):**
```
L_box = 1 - CIoU(B_pred, B_gt)

CIoU = IoU - (ρ²(b, b_gt) / c²) - α * v

where:
  IoU = |B_pred ∩ B_gt| / |B_pred ∪ B_gt|
  ρ = Euclidean distance between centers
  c = diagonal length of smallest enclosing box
  v = (4/π²) * (arctan(w_gt/h_gt) - arctan(w/h))²
  α = v / (1 - IoU + v)
```

**2. Classification Loss:**
```
L_cls = -Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

Binary cross-entropy for each class

**3. Distribution Focal Loss:**
```
L_dfl = -Σ [(y_left * log(p_left) + y_right * log(p_right))]
```

For bbox coordinate regression

#### 3.3.3 Optimization

**Optimizer: SGD (Stochastic Gradient Descent)**
```
θ_t+1 = θ_t - η * ∇L(θ_t) + μ * v_t

where:
  η = 0.01 (learning rate)
  μ = 0.937 (momentum)
  weight_decay = 0.0005
```

**Learning Rate Schedule (Cosine Annealing):**
```
η_t = η_min + (η_max - η_min) * (1 + cos(π * t / T)) / 2

where:
  η_max = 0.01
  η_min = 0.0001
  T = 100 (total epochs)
```

**Warmup (First 3 Epochs):**
```
η_t = η_base * (t / T_warmup)

where t ∈ [0, 3]
```

#### 3.3.4 Evaluation Metrics

**Mean Average Precision at IoU=0.5:**
```
AP = ∫₀¹ P(R) dR

mAP@0.5 = (1/N) * Σ AP_i

where:
  P = TP / (TP + FP)  (Precision)
  R = TP / (TP + FN)  (Recall)
  N = number of classes
```

**Mean Average Precision at IoU=[0.5:0.95]:**
```
mAP@0.5:0.95 = (1/10) * Σ(IoU=0.5 to 0.95, step=0.05) mAP@IoU
```

---

### 3.4 Video Inference with ByteTrack

#### 3.4.1 Detection Phase

For each frame f_t:
```
D_t = YOLOv8(f_t) = {(b_i, c_i, s_i) | i = 1...N}

where:
  b_i = bounding box [x1, y1, x2, y2]
  c_i = class ID
  s_i = confidence score
```

#### 3.4.2 ByteTrack Algorithm

**Step 1: High-Confidence Track Association**

For detections with s_i > τ_high (τ_high = 0.4):

```
IoU(b_det, b_track) = Area(b_det ∩ b_track) / Area(b_det ∪ b_track)

Match if: IoU(b_det, b_track) > τ_match (τ_match = 0.8)
```

**Step 2: Low-Confidence Re-identification**

For detections with τ_low < s_i < τ_high (τ_low = 0.1):

```
Match with lost tracks if:
  IoU(b_det, b_lost) > τ_rematch (τ_rematch = 0.5)
```

**Step 3: Track Management**

```
lost_frames_i = {
  0                    if matched at frame t
  lost_frames_i + 1    if not matched
}

Remove track if: lost_frames_i > buffer_size (buffer_size = 30)
```

#### 3.4.3 Temporal Smoothing

**Moving Average Filter:**

For track trajectory T = {b_1, b_2, ..., b_t}:

```
b_smooth(t) = (1/w) * Σ(i=t-w+1 to t) b_i

where w = 5 (window size)
```

**Component-wise:**
```
x1_smooth = mean([x1_(t-4), x1_(t-3), x1_(t-2), x1_(t-1), x1_t])
y1_smooth = mean([y1_(t-4), y1_(t-3), y1_(t-2), y1_(t-1), y1_t])
x2_smooth = mean([x2_(t-4), x2_(t-3), x2_(t-2), x2_(t-1), x2_t])
y2_smooth = mean([y2_(t-4), y2_(t-3), y2_(t-2), y2_(t-1), y2_t])
```

**Benefits:**
- Reduces jitter in bbox coordinates
- Stabilizes tracking visualization
- Smooths motion trajectory

---

### 3.5 Spatio-Temporal IoU Metric

#### 3.5.1 Definition

Given predictions P and ground truth G for video V:

```
ST-IoU(P, G) = Σ(f ∈ F_intersect) IoU(B_f^P, B_f^G) / |F_union|

where:
  F_intersect = {f | f has both prediction and ground truth}
  F_union = {f | f has prediction OR ground truth}
  B_f^P = predicted bbox at frame f
  B_f^G = ground truth bbox at frame f
```

#### 3.5.2 Bbox Matching

For multiple objects in frame f:

**Hungarian Algorithm (Optimal Matching):**

```
Cost Matrix: C[i,j] = 1 - IoU(B_i^P, B_j^G)

Minimize: Σ C[i, π(i)]

where π is a permutation (matching)
```

**Simplified Greedy Matching (Implemented):**

```
1. Compute IoU matrix M[i,j] for all prediction-GT pairs
2. While unmatched pairs exist:
   a. Find max IoU in M
   b. If max IoU > threshold:
      - Match pair (i*, j*)
      - Remove row i* and column j*
   c. Else: break
```

#### 3.5.3 Example Calculation

**Given:**
- Ground Truth: frames {100, 101, 102}
- Prediction: frames {101, 102, 103}

**Compute:**

Frame 101:
```
GT:   [50, 50, 100, 100]
Pred: [52, 48, 98, 102]

Intersection = (98-52) * (100-48) = 46 * 52 = 2392
Union = 2500 + 2496 - 2392 = 2604
IoU_101 = 2392 / 2604 = 0.919
```

Frame 102:
```
GT:   [55, 55, 105, 105]
Pred: [54, 56, 104, 106]

IoU_102 = 0.890
```

**Final ST-IoU:**
```
F_intersect = {101, 102}
F_union = {100, 101, 102, 103}

ST-IoU = (0.919 + 0.890) / 4 = 0.452
```

---

### 3.6 TensorRT Optimization (Jetson Deployment)

#### 3.6.1 Model Conversion Pipeline

```
PyTorch (.pt)
    |
    | [export to ONNX]
    v
ONNX (.onnx)
    |
    | [TensorRT builder]
    v
TensorRT Engine (.engine)
```

#### 3.6.2 FP16 Quantization

**Precision Reduction:**
```
FP32: 32 bits (1 sign, 8 exponent, 23 mantissa)
FP16: 16 bits (1 sign, 5 exponent, 10 mantissa)
```

**Benefits:**
- 2x memory reduction
- 2-3x inference speedup
- Minimal accuracy loss (< 1% mAP degradation)

#### 3.6.3 Performance Targets

```
Platform: NVIDIA Jetson Xavier NX

Target Metrics:
- Throughput: >= 15 FPS
- Latency: <= 66 ms/frame
- Memory: <= 8 GB
- Power: <= 15W
```

---

## 4. EXPERIMENTAL SETUP

### 4.1 Dataset Statistics

```
Training Set:
- Videos: 11
- Total Frames: ~2,200-3,300
- Objects per Frame: 1-3
- Classes: 7 (Backpack, Jacket, Laptop, Lifering, MobilePhone, Person1, WaterBottle)

Validation Set:
- Videos: 3
- Total Frames: ~600-900
```

### 4.2 Training Configuration

```
Model: YOLOv8n
Input Size: 640x640
Batch Size: 16
Epochs: 100
Early Stopping: patience=20

Augmentation:
- Mosaic: 1.0
- MixUp: 0.2
- Rotation: ±10°
- HSV: (0.015, 0.7, 0.4)
- Flip Horizontal: 0.5
```

### 4.3 Inference Configuration

```
Detection:
- Confidence Threshold: 0.25
- NMS IoU Threshold: 0.45

Tracking:
- Track Threshold: 0.4
- Track Buffer: 30 frames
- Match Threshold: 0.8
- Smooth Window: 5 frames
```

---
