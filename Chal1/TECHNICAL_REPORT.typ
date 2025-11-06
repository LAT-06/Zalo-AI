= System Architecture

== Pipeline Overview

The complete pipeline consists of five main stages:

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [*Stage*], [*Description*],
  [1], [Dataset Parsing & Preprocessing],
  [2], [Data Augmentation Pipeline],
  [3], [YOLOv8 Training],
  [4], [Video Inference + ByteTrack],
  [5], [ST-IoU Evaluation & Submission],
)

#v(1cm)

```
Input: Drone Videos + Annotations
    ↓
[STEP 1] Dataset Parsing
    ↓
[STEP 2] Augmentation
    ↓
[STEP 3] Training
    ↓
[STEP 4] Inference
    ↓
[STEP 5] Evaluation
    ↓
Output: submission.json
```

= Methodology

== Dataset Parsing

=== Coordinate Transformation

Given bounding box coordinates $(x_1, y_1, x_2, y_2)$ and image dimensions $(W, H)$, we transform to YOLO format:

$ x_"center" = (x_1 + x_2) / (2W) $

$ y_"center" = (y_1 + y_2) / (2H) $

$ w = (x_2 - x_1) / W $

$ h = (y_2 - y_1) / H $

where all values are normalized to the range $[0, 1]$.

=== Frame Sampling Strategy

To reduce computational cost while maintaining detection coverage:
- Extract all frames with ground truth annotations
- Sample negative frames (no objects) every 50 frames
- Result: approximately 200-300 frames per video

=== Train/Validation Split

- Split ratio: 80% training, 20% validation
- 14 videos total → 11 training, 3 validation
- Random split (no stratification due to small dataset size)

== Data Augmentation

=== Mosaic Augmentation

Mosaic combines 4 images into a single training sample. For image $i$ in quadrant $q$, the new bounding box coordinates are:

$ x_"new" = (x_"old" times s_i + "offset"_x^q) / W_"mosaic" $

$ y_"new" = (y_"old" times s_i + "offset"_y^q) / H_"mosaic" $

where $s_i$ is the scaling factor for image $i$.

*Benefits:*
- Increases batch diversity
- Improves small object detection
- Simulates multiple objects in scene

=== Random Scaling

Scale range: $s in [0.5, 1.5]$

For image $I$ with dimensions $(W, H)$:

$ W_"new" = W times s $
$ H_"new" = H times s $

Corresponding bounding box transformation:

$ x_"center"^"new" = (x_"center" times s) / W_"new" $
$ y_"center"^"new" = (y_"center" times s) / H_"new" $

=== Rotation

Rotation angle: $theta in [-10°, +10°]$

*Why limited rotation?*
- Aerial perspective constraint: sky always up, ground always down
- Large rotations break semantic meaning
- Small rotations simulate drone tilt

Rotation matrix:

$ bold(R)(theta) = mat(
  cos(theta), -sin(theta);
  sin(theta), cos(theta)
) $

=== HSV Color Jitter

Transformation parameters:

$ H_"new" = H + delta_h times 180° quad "where" delta_h in [-0.015, 0.015] $

$ S_"new" = S times (1 + delta_s) quad "where" delta_s in [-0.7, 0.7] $

$ V_"new" = V times (1 + delta_v) quad "where" delta_v in [-0.4, 0.4] $

*Purpose:* Simulate varying lighting conditions (dawn, noon, dusk, cloudy)

=== MixUp Augmentation

MixUp creates soft samples by linear interpolation:

$ x_"mixed" = lambda x_i + (1 - lambda) x_j $

$ y_"mixed" = {y_i union y_j} $

where $lambda tilde "Beta"(alpha, alpha)$ with $alpha = 0.2$


== YOLOv8 Training

=== Model Architecture

*Backbone:* CSPDarknet
- Cross Stage Partial connections
- Reduces computational redundancy
- Maintains gradient flow

*Neck:* Path Aggregation Network (PAN)
- Bottom-up and top-down feature fusion
- Multi-scale feature extraction

*Head:* Decoupled Detection Head
- Separate branches for classification and localization
- Anchor-free detection

=== Loss Function

The total loss is a weighted combination of three components:

$ cal(L)_"total" = lambda_"box" cal(L)_"box" + lambda_"cls" cal(L)_"cls" + lambda_"dfl" cal(L)_"dfl" $

where $lambda_"box" = 7.5$, $lambda_"cls" = 0.5$, $lambda_"dfl" = 1.5$

==== Box Loss (Complete IoU)

$ cal(L)_"box" = 1 - "CIoU"(B_"pred", B_"gt") $

$ "CIoU" = "IoU" - rho^2(b, b_"gt") / c^2 - alpha v $

where:
- $"IoU" = |B_"pred" inter B_"gt"| / |B_"pred" union B_"gt"|$
- $rho$ = Euclidean distance between centers
- $c$ = diagonal length of smallest enclosing box
- $v = (4/pi^2)(arctan(w_"gt"/h_"gt") - arctan(w/h))^2$
- $alpha = v / (1 - "IoU" + v)$

==== Classification Loss

Binary cross-entropy for each class:

$ cal(L)_"cls" = -sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)] $

==== Distribution Focal Loss

For bounding box coordinate regression:

$ cal(L)_"dfl" = -sum [(y_"left" log(p_"left") + y_"right" log(p_"right"))] $

=== Optimization

*Optimizer:* Stochastic Gradient Descent (SGD)

$ theta_(t+1) = theta_t - eta nabla cal(L)(theta_t) + mu v_t $

where:
- $eta = 0.01$ (learning rate)
- $mu = 0.937$ (momentum)
- weight decay $= 0.0005$

*Learning Rate Schedule:* Cosine Annealing

$ eta_t = eta_"min" + (eta_"max" - eta_"min") (1 + cos(pi t / T)) / 2 $

where $eta_"max" = 0.01$, $eta_"min" = 0.0001$, $T = 100$ epochs

*Warmup:* First 3 epochs

$ eta_t = eta_"base" times (t / T_"warmup") quad "for" t in [0, 3] $

=== Evaluation Metrics

*Mean Average Precision at IoU=0.5:*

$ "AP" = integral_0^1 P(R) dif R $

$ "mAP"\@0.5 = 1/N sum_i "AP"_i $

where:
- $P = "TP" / ("TP" + "FP")$ (Precision)
- $R = "TP" / ("TP" + "FN")$ (Recall)
- $N$ = number of classes

*Mean Average Precision at IoU=[0.5:0.95]:*

$ "mAP"\@[0.5:0.95] = 1/10 sum_("IoU"=0.5)^0.95 "mAP"\@"IoU" $
==
== Video Inference with ByteTrack

=== Detection Phase

For each frame $f_t$, apply YOLOv8:

$ cal(D)_t = "YOLOv8"(f_t) = {(b_i, c_i, s_i) | i = 1...N} $

where:
- $b_i$ = bounding box $[x_1, y_1, x_2, y_2]$
- $c_i$ = class ID
- $s_i$ = confidence score

=== ByteTrack Algorithm

*Step 1: High-Confidence Track Association*

For detections with $s_i > tau_"high"$ ($tau_"high" = 0.4$):

$ "IoU"(b_"det", b_"track") = "Area"(b_"det" inter b_"track") / "Area"(b_"det" union b_"track") $

Match if $"IoU"(b_"det", b_"track") > tau_"match"$ ($tau_"match" = 0.8$)

*Step 2: Low-Confidence Re-identification*

For detections with $tau_"low" < s_i < tau_"high"$ ($tau_"low" = 0.1$):

Match with lost tracks if $"IoU"(b_"det", b_"lost") > tau_"rematch"$ ($tau_"rematch" = 0.5$)

*Step 3: Track Management*

$ "lost_frames"_i = cases(
  0 quad &"if matched at frame" t,
  "lost_frames"_i + 1 quad &"if not matched"
) $

Remove track if $"lost_frames"_i > "buffer_size"$ ($"buffer_size" = 30$)

=== Temporal Smoothing

*Moving Average Filter:*

For track trajectory $cal(T) = {b_1, b_2, ..., b_t}$:

$ b_"smooth"(t) = 1/w sum_(i=t-w+1)^t b_i $

where $w = 5$ (window size)

Component-wise smoothing:

$ x_1^"smooth" = 1/5 sum_(i=t-4)^t x_1^i $

Benefits:
- Reduces jitter in bounding box coordinates
- Stabilizes tracking visualization
- Smooths motion trajectory

== Spatio-Temporal IoU Metric

=== Definition

Given predictions $cal(P)$ and ground truth $cal(G)$ for video $cal(V)$:

$ "ST-IoU"(cal(P), cal(G)) = (sum_(f in cal(F)_"intersect") "IoU"(B_f^cal(P), B_f^cal(G))) / |cal(F)_"union"| $

where:
- $cal(F)_"intersect"$ = frames with both prediction and ground truth
- $cal(F)_"union"$ = frames with prediction OR ground truth
- $B_f^cal(P)$ = predicted bbox at frame $f$
- $B_f^cal(G)$ = ground truth bbox at frame $f$

=== Bounding Box Matching

For multiple objects in frame $f$, use greedy matching:

1. Compute IoU matrix $bold(M)[i,j]$ for all prediction-GT pairs
2. While unmatched pairs exist:
   - Find $max "IoU"$ in $bold(M)$
   - If $max "IoU" > "threshold"$:
     - Match pair $(i^*, j^*)$
     - Remove row $i^*$ and column $j^*$
   - Else: break

=== Example Calculation

*Given:*
- Ground Truth: frames ${100, 101, 102}$
- Prediction: frames ${101, 102, 103}$

*Frame 101:*
- GT: $[50, 50, 100, 100]$
- Pred: $[52, 48, 98, 102]$
- Intersection $= (98-52) times (100-48) = 2392$
- Union $= 2500 + 2496 - 2392 = 2604$
- $"IoU"_101 = 2392 / 2604 = 0.919$

*Frame 102:*
- $"IoU"_102 = 0.890$

*Final ST-IoU:*

$ cal(F)_"intersect" = {101, 102} $
$ cal(F)_"union" = {100, 101, 102, 103} $
$ "ST-IoU" = (0.919 + 0.890) / 4 = 0.452 $
#pagebreak()

== TensorRT Optimization

=== Model Conversion Pipeline

```
PyTorch (.pt)
    ↓ [export to ONNX]
ONNX (.onnx)
    ↓ [TensorRT builder]
TensorRT Engine (.engine)
```
