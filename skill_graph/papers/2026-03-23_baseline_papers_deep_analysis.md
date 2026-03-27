# Baseline Papers Deep Technical Analysis — 2026-03-23

> **구분**: baseline / competitor
> **우선순위**: high
> **keywords**: VAD, anomaly detection, reconstruction, memory bank, CLIP, residual features, universal AD

---

## 1. Dinomaly (CVPR 2025)

**Paper**: "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection"
**Code**: https://github.com/guojiajeremy/Dinomaly
**Link**: https://arxiv.org/abs/2405.14325

### 1.1 Architecture

```
[Input Image]
    |
[Encoder] DINOv2-Reg ViT-Base/14 (frozen)
    | Extract features from layers [2,3,4,5,6,7,8,9] (Base) or [4,6,8,10,12,14,16,18] (Large)
    |
[Noisy Bottleneck] MLP with Dropout (p=0.2, or 0.4 for Real-IAD)
    | Collects multi-layer features, compresses them
    |
[Decoder] 8 Transformer layers with LINEAR Attention (not Softmax)
    | Reconstructs features
    |
[Loose Reconstruction Loss] Grouped-layer cosine distance + hard mining
    |
[Anomaly Score] = Reconstruction error (cosine distance)
```

**Encoder**: Pre-trained DINOv2 with registers (DINOv2-R), ViT-Base/14 by default. Completely frozen during training. Features extracted from 8 intermediate layers (middle layers, not first/last).

**Noisy Bottleneck**: Simple MLP that fuses multi-layer encoder features. The key "noise" is just standard Dropout (p=0.2 default). This prevents the decoder from learning identity mapping for both normal and anomalous patterns. During training, random feature dimensions are zeroed out, forcing the decoder to reconstruct from incomplete information.

**Decoder**: 8 Transformer layers using **Linear Attention** instead of Softmax Attention. This is a deliberate architectural choice, not just for efficiency.

**Optimizer**: StableAdamW with warm cosine LR scheduling.

### 1.2 The Four Essential Components (with rationale)

**Component 1: Foundation Transformers (DINOv2)**
- Self-supervised ViTs provide universal, discriminative features across diverse object classes
- Scaling law observed: larger ViTs -> better multi-class AD
- DINOv2 with registers (DINOv2-R) avoids artifact tokens that harm reconstruction

**Component 2: Noisy Bottleneck (Dropout)**
- Addresses "over-generalization" problem: in multi-class settings, diverse training patterns cause decoders to generalize reconstruction to unseen anomalous patterns too
- Dropout randomly zeros feature dimensions during training -> decoder cannot rely on complete information -> cannot perfectly reconstruct unseen anomalous patterns
- Default dropout rate: 0.2 (increased to 0.4 for diverse Real-IAD with 30 classes)
- This is elegantly simple: no learned noise, no VAE-style reparameterization, just standard Dropout

**Component 3: Linear Attention**
- Softmax Attention concentrates on local regions matching the query -> can copy identical anomalous patterns through
- Linear Attention (Q*K^T replaced with phi(Q)*phi(K)^T) naturally spreads attention globally -> CANNOT focus on exact local regions
- This "side effect" is exploited: the decoder must use global context for reconstruction, making it harder to forward anomalous local patterns
- Computational bonus: O(N*d^2) vs O(N^2*d) for Softmax

**Component 4: Loose Reconstruction**
- Traditional: force layer-i decoder output to match layer-i encoder output (strict layer-to-layer)
- Dinomaly: group multiple encoder layers and match them to grouped decoder outputs
- Uses progressive hard mining: percentage of hardest features increases from 0 to 0.9 over 1000 steps
- Cosine similarity loss, but discards well-reconstructed regions during optimization
- Prevents the model from wasting capacity on easy reconstructions

### 1.3 Results

| Dataset | Setting | I-AUROC | Notes |
|---------|---------|---------|-------|
| MVTec AD | Multi-class (Base) | 99.6% | |
| MVTec AD | Multi-class (Large) | 99.8% | |
| VisA | Multi-class (Base) | 98.7% | |
| VisA | Multi-class (Large) | 98.9% | |
| Real-IAD | Multi-class (Base) | 89.3% | Dropout 0.4 |
| Real-IAD | Multi-class (Large) | 90.1% | |

**Key achievement**: First multi-class UAD model that competes with per-class SOTAs.

### 1.4 Failure Modes and Limitations

1. **Real-IAD performance gap**: On the complex Real-IAD (30 classes, multi-view), multi-class Dinomaly still suffers moderate drop vs per-class. The diversity ceiling hasn't been fully overcome.
2. **Over-generalization not fully solved**: The four components mitigate but don't eliminate the fundamental tension between reconstruction quality for normals and suppression for anomalies.
3. **Foundation model dependency**: Performance is tightly coupled to DINOv2 quality. No analysis of what happens when DINOv2 features are suboptimal (e.g., for domains far from ImageNet).
4. **No explicit handling of distribution shift**: The paper doesn't evaluate robustness to lighting/environmental changes (no MVTec AD 2 evaluation).
5. **Logical anomalies**: As a reconstruction-based method, it fundamentally detects structural/textural anomalies. Logical anomalies (wrong count, wrong arrangement) are likely missed since individual patches still look normal.

---

## 2. AnomalyCLIP (ICLR 2024)

**Paper**: "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection"
**Code**: https://github.com/zqhang/AnomalyCLIP
**Link**: https://arxiv.org/abs/2310.18961

### 2.1 Architecture

```
[Input Image]
    |
[CLIP Visual Encoder] ViT-L/14@336px (frozen)
    | Extract local features from layers 6, 12, 18, 24
    | Apply DPAM (V-V attention) starting from layer 6
    |
[CLIP Text Encoder] (frozen backbone + learnable prompt tokens)
    | Learnable prompt length = 12 tokens
    | Inserted into first 9 layers of text encoder
    | Two prompts: [normal] and [anomalous]
    |
[Matching]
    | Cosine similarity between visual features and text embeddings
    | Global (image-level) + Local (pixel-level) matching
    |
[Anomaly Map] = 1 - similarity(visual, normal_prompt) weighted with similarity(visual, anomalous_prompt)
```

### 2.2 Key Technical Details

**Object-Agnostic Prompts**: Instead of "a photo of a damaged [object]", AnomalyCLIP learns prompts like "a photo of a damaged []" where [] are learnable tokens. This removes object-specific semantics, allowing the model to focus on normality/abnormality patterns regardless of object type.

**DPAM (Diagonally Prominent Attention Maps)**:
- Standard CLIP visual attention focuses on object-level semantics (what class of object)
- DPAM introduces V-V (value-value) self-attention that creates diagonally prominent attention patterns
- This forces the model to attend to local patch-level features rather than global object semantics
- Applied from the 6th layer onward in the visual encoder
- Critical for pixel-level anomaly segmentation

**Training Procedure**:
- Backbone: CLIP ViT-L/14@336px, ALL CLIP parameters frozen
- Only learnable: prompt tokens + DPAM layers + alignment projections
- Training data: auxiliary AD dataset (e.g., MVTec AD training split as proxy)
- Optimizer: AdamW
- Number of learnable text tokens per layer: 4 (inserted into 9 layers)
- Total prompt length: 12 learnable tokens
- Two sets of prompts learned: one for "normal", one for "anomalous"

**Multi-Scale Fusion**: Local visual features from layers 6, 12, 18, 24 are combined for pixel-level scoring.

### 2.3 The 17 Evaluation Datasets

**Industrial (8)**: MVTec AD, VisA, MPDD, BTAD, SDD, DAGM, DTD-Synthetic, (+ one more)
**Medical (9+)**: HeadCT, BrainMRI, Br35H, COVID-19, ISIC, CVC-ColonDB, CVC-ClinicDB, Kvasir, Endo, TN3K

### 2.4 Results (representative)

| Dataset | Domain | Image AUROC | Pixel AUROC |
|---------|--------|-------------|-------------|
| MVTec AD | Industrial | ~91.5% | ~95.5% |
| VisA | Industrial | ~82.7% | ~96.2% |
| MPDD | Industrial | ~77.0% | ~95.0% |
| BTAD | Industrial | ~84.0% | ~95.0% |

(These are zero-shot results — no training on target domain data)

### 2.5 Failure Modes and Limitations

1. **Medical domain gap**: Shows consistently low Dice Scores when transferring from industrial to medical domains. The "object-agnostic" claim doesn't fully hold across dramatically different domains.
2. **Image-level detection weakness**: Zero-shot image AUROC is significantly lower than supervised methods (91.5% vs 99.6% for Dinomaly on MVTec). The gap is especially large for subtle anomalies.
3. **Per-category variance**: High variance across categories. Performs well on textured objects but struggles with semantic/logical anomalies.
4. **Prompt sensitivity**: Performance depends on auxiliary training data choice. If proxy training data doesn't cover the diversity of target anomalies, generalization suffers.
5. **Resolution limitation**: CLIP's 336x336 input resolution limits detection of small defects common in industrial settings.
6. **No fine-grained localization**: Pixel-level maps are coarse due to CLIP's patch-level (14x14 grid at 336px) feature resolution.

---

## 3. PatchCore (CVPR 2022)

**Paper**: "Towards Total Recall in Industrial Anomaly Detection"
**Code**: https://github.com/amazon-science/patchcore-inspection
**Link**: https://arxiv.org/abs/2106.08265

### 3.1 Core Algorithm

```
=== Training ===
[Normal Images]
    |
[Pre-trained CNN] WideResNet-101 (frozen), extract from layers 2 and 3
    |
[Local Neighborhood Aggregation] For each patch at (h,w), aggregate features from p-neighborhood
    | Increases receptive field without going deeper (avoids ImageNet bias)
    |
[Memory Bank] M = {all aggregated patch features from all normal training images}
    | Size: N_images * H * W feature vectors
    |
[Coreset Subsampling] Greedy minimax facility location
    | Iteratively select patches maximally distant from already-selected set
    | Retain 1-25% of original bank (default ~10%)
    |
[Reduced Memory Bank] M_c (compact)

=== Inference ===
[Test Image]
    |
[Same Feature Extraction + Aggregation]
    |
[k-NN Scoring] For each test patch, find nearest neighbor in M_c
    | Pixel-level score = distance to nearest neighbor
    | Image-level score = max over all pixel scores
    |   + re-weighting: score *= (# neighbors within threshold) / softmax normalization
```

### 3.2 Key Technical Details

**Feature Extraction**: Mid-level CNN features (layers 2-3) are deliberately chosen over deeper layers. Deeper layers carry more ImageNet class semantics (bias), while mid-level layers capture textures and shapes more relevant to anomaly detection.

**Neighborhood Aggregation**: For spatial position (h,w), features from a local neighborhood of size p are aggregated (average pooled). This provides robustness to small spatial misalignments while preserving resolution.

**Coreset Selection (Minimax Facility Location)**:
- Goal: select subset S from M such that max_{m in M} min_{s in S} d(m,s) is minimized
- Greedy algorithm: iteratively pick the point that is farthest from the current coreset
- Approximation of K-center covering problem
- Retaining 1% of features: ~negligible AUROC loss, 100x inference speedup

**Scoring**: Not just nearest-neighbor distance. Includes a re-weighting factor based on local density in the memory bank neighborhood. This improves discrimination.

### 3.3 Results

| Dataset | Subsampling | I-AUROC | P-AUROC | P-PRO |
|---------|-------------|---------|---------|-------|
| MVTec AD | 25% | 99.1% | 98.1% | 93.5% |
| MVTec AD | 10% | ~99.0% | ~98.0% | ~93.3% |
| MVTec AD | 1% | ~98.7% | ~97.8% | ~93.0% |

### 3.4 Why Robust to Distribution Shift (MVTec AD 2)?

On MVTec AD 2, PatchCore shows only ~3pp drop under lighting changes (vs >10pp for EfficientAD). Reasons:

1. **Non-parametric nature**: No learned decision boundary that can overfit to training conditions. The memory bank simply stores "what normal looks like" and measures distance.
2. **Mid-level features**: CNN layers 2-3 features are more robust to illumination changes than deeper features (which encode more scene-level information).
3. **Pre-trained features**: Using frozen ImageNet features that have already seen diverse conditions provides inherent robustness.
4. **Distance-based scoring**: k-NN scoring gracefully degrades — a lighting shift moves features slightly, but the relative distance ordering is more preserved than for discriminative boundaries.

### 3.5 Memory Requirements and Scalability

- Memory bank size: O(N_images * resolution^2 * feature_dim)
- With coreset 1%: practical for moderate-scale deployment
- Inference: dominated by nearest-neighbor search in memory bank
- For large-scale: requires FAISS or approximate NN search
- Multi-class: requires separate memory banks per class (no unified model)

### 3.6 Known Limitations

1. **Spatial misalignment sensitivity**: Despite neighborhood aggregation, significant rotation/flipping/translation degrades performance. Feature matching assumes approximate spatial correspondence.
2. **No multi-class unification**: Requires separate memory bank per class. Cannot handle mixed classes in a single model.
3. **Memory scaling**: For large datasets or high-resolution images, memory bank grows prohibitively even with coreset selection.
4. **ImageNet bias**: Pre-trained features may not capture domain-specific patterns (e.g., semiconductor defects, medical images).
5. **Noise vulnerability**: 18.58% AUROC drop under Gaussian noise. Memory-based methods are overconfident in training data distribution.
6. **Background sensitivity**: Cluttered or non-standard backgrounds degrade both detection and localization.
7. **Fine-grained anomalies**: Subtle texture changes within similar-looking patches can be missed.
8. **No logical anomaly detection**: Cannot detect structural/positional anomalies (wrong count, wrong arrangement).

---

## 4. ResAD (NeurIPS 2024 Spotlight)

**Paper**: "ResAD: A Simple Framework for Class Generalizable Anomaly Detection"
**Code**: https://github.com/xcyao00/ResAD
**Link**: https://arxiv.org/abs/2410.20047

### 4.1 Architecture

```
=== Setup (per-class few-shot reference) ===
[Few-shot Normal Images per class] (e.g., 4-shot)
    |
[Pre-trained Feature Extractor] (e.g., WideResNet-50)
    |
[Normal Reference Feature Pool] R = {features from few-shot normals}

=== Feature Conversion ===
[Input Image]
    |
[Pre-trained Feature Extractor] Same backbone
    | Initial features F
    |
[Nearest Neighbor Matching] For each feature f_i, find nearest r_j in R
    |
[Residual Computation] residual_i = f_i - r_j
    | Key insight: residual features have MUCH lower cross-class variation

=== Feature Constraintor ===
[Residual Features]
    |
[Shallow MLP Network]
    | Projects residual features onto spatial hypersphere
    | Trained with Abnormal Invariant OCC Loss:
    |   - Normal residuals -> compact cluster (small norm)
    |   - Anomalous residuals -> preserved (not collapsed)
    |
[Constrained Residual Features] Normalized, class-consistent scale

=== Feature Distribution Estimator ===
[Constrained Residual Features]
    |
[Normalizing Flow] Real-NVP architecture
    | Coupling layers with 2-layer MLP subnets (replacing conv subnets)
    | Trained with maximum likelihood (log-likelihood optimization)
    |
[Anomaly Score] = negative log-likelihood
```

### 4.2 Why Residual Features Reduce Cross-Class Variation

The core insight is elegant:
- **Initial features**: A bottle feature and a screw feature are VERY different (high inter-class variation)
- **After NN matching**: Each feature is matched to its nearest normal reference, so the "baseline" is already class-specific
- **Residual = deviation from normal**: For normal samples of ANY class, the residual is small and similar (close to zero). For anomalous regions, the residual is large regardless of class.
- **Result**: The residual feature distribution is class-invariant for normals, while anomalous residuals remain distinguishable

This is analogous to how background subtraction works in video surveillance — subtract the "expected" and the residual contains the anomalies.

### 4.3 Component Details

**Feature Converter**: Not a learned module — it's nearest-neighbor lookup + subtraction. The "conversion" is the subtraction of matched normal references.

**Feature Constraintor**: Shallow MLP that projects onto a hypersphere.
- Abnormal Invariant OCC Loss: Makes normal residuals compact while preserving anomalous residual characteristics
- Ensures cross-class feature scale consistency (important because different classes may have different raw feature magnitudes)

**Feature Distribution Estimator**: Real-NVP normalizing flow
- Coupling layers with 2-layer MLP subnets
- Estimates p(residual | normal) via change-of-variables
- Anomaly score = negative log-likelihood

### 4.4 Results

| Train -> Test | Setting | I-AUROC | P-AUROC |
|--------------|---------|---------|---------|
| MVTecAD (self) | Per-class | 90.5% | 95.7% |
| MVTecAD (w/o residual) | Per-class | 72.8% | 82.9% |
| MVTecAD -> BraTS | Cross-domain, 4-shot | 84.6% | 96.1% |
| MVTecAD -> ShanghaiTech | Cross-domain, 4-shot | 84.3% | 92.6% |

**Ablation highlight**: Removing residual feature learning causes a dramatic 17.7pp drop in I-AUROC (90.5 -> 72.8), confirming it is the critical component.

Best or second-best in 13 out of 15 MVTec categories.

### 4.5 Limitations

1. **Few-shot reference dependency**: Requires few-shot normal samples per target class. Not truly zero-shot.
2. **NN matching quality**: If the reference pool is too small or unrepresentative, the matched normal reference may be poor, leading to noisy residuals.
3. **Static reference pool**: Cannot adapt to distribution drift in the normal data over time.
4. **NF capacity**: Real-NVP with MLP subnets may underfit complex residual distributions for highly variable classes.
5. **Per-class reference needed**: While the model is class-generalizable, you still need class-specific reference images at test time.

---

## 5. EfficientAD (WACV 2024)

**Paper**: "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"
**Code**: https://github.com/nelson1425/EfficientAD (unofficial)
**Link**: https://arxiv.org/abs/2303.14535

### 5.1 Architecture

```
=== Teacher (Pre-distilled PDN) ===
[Pre-trained WideResNet-101 features]
    | Distilled into lightweight PDN
    |
[Patch Description Network (PDN)] 4 conv layers
    | Layer 1: Conv2d(3, 128, k=4) + AvgPool
    | Layer 2: Conv2d(128, 256, k=4) + AvgPool
    | Layer 3: Conv2d(256, 256, k=3)
    | Layer 4: Conv2d(256, 384, k=4)
    | Receptive field: 33x33 pixels per output neuron
    | Trained on ImageNet to mimic WideResNet-101 features

=== Student (Same PDN architecture) ===
[Student PDN] Same architecture as teacher
    | Trained to predict teacher output ON NORMAL IMAGES ONLY
    | + Penalty loss: prevents student from generalizing beyond normals
    |
[Structural Anomaly Score] = ||Teacher(x) - Student(x)||^2

=== Autoencoder (for logical anomalies) ===
[Convolutional Autoencoder]
    | Encoder: strided convolutions (downsample)
    | Decoder: bilinear upsampling
    | Trained to reconstruct teacher features
    | Student also predicts AE output
    |
[Logical Anomaly Score] = ||AE(x) - Student_AE(x)||^2

=== Final Score ===
Anomaly Score = Structural Score + Logical Score
```

### 5.2 Why It's Fast

- **Only 4 conv layers**: PDN has minimal depth. Each output describes a 33x33 patch.
- **No transformer, no attention**: Pure convolutional, highly parallelizable.
- **No memory bank**: Unlike PatchCore, no NN search at inference.
- **Strided average pooling**: Reduces spatial dimensions early.
- **Result**: <2ms latency, >600 images/second on GPU.

### 5.3 Key Training Details

- Teacher PDN is pre-trained by distilling WideResNet-101 on ImageNet
- Student PDN trained for ~70,000 steps on normal training images
- Novel penalty loss: the student is penalized for producing outputs similar to the teacher on ImageNet images (non-target-domain images). This prevents the student from learning a general feature extractor and forces it to specialize on normal patterns of the target class.
- Autoencoder trains concurrently, with student additionally predicting AE output.

### 5.4 Results

| Dataset | Variant | I-AUROC | Notes |
|---------|---------|---------|-------|
| MVTec AD | EfficientAD-M | 99.8% | |
| MVTec AD | EfficientAD-S | 99.1% | |
| VisA | EfficientAD-M | 98.1% | |
| VisA | EfficientAD-S | 97.5% | |
| MVTec LOCO | EfficientAD-M | ~90% | Logical anomalies |

### 5.5 Why NOT Robust to Distribution Shift (MVTec AD 2)

On MVTec AD 2, EfficientAD shows **>10pp AU-PRO drop** under lighting changes, while PatchCore/RD drop only ~3pp.

**Root causes of fragility**:

1. **Learned decision boundary**: The student learns to exactly match the teacher on normal data. Any shift in input distribution (lighting change) causes BOTH teacher and student outputs to change, but differently — the student has only seen normal data under one lighting condition. The discrepancy becomes unreliable.

2. **Shallow architecture**: With only 4 conv layers and 33x33 receptive field, the PDN is very sensitive to low-level appearance changes. It lacks the depth to learn illumination-invariant features.

3. **Penalty on ImageNet**: The penalty term forces the student to NOT generalize. While this prevents false negatives on anomalies, it also makes the student brittle — it can only handle the exact distribution seen during training.

4. **No explicit invariance**: No data augmentation for lighting, no normalization layers designed for illumination robustness. The speed comes at the cost of robustness.

5. **Autoencoder limitations**: The convolutional AE also learns a fixed reconstruction distribution. Lighting changes create reconstruction errors that look like anomalies.

**In essence**: EfficientAD achieves speed by being extremely specialized to the training distribution. This specialization is precisely what makes it fragile to distribution shift.

### 5.6 MVTec AD 2 Comparison

| Method | Regular AU-PRO | With Lighting Change | Drop |
|--------|---------------|---------------------|------|
| EfficientAD | 30.8% | ~20% | >10pp |
| PatchCore | ~28% | ~25% | ~3pp |
| RD | ~27% | ~24% | ~3pp |

(Note: MVTec AD 2 is significantly harder than MVTec AD — no method exceeds ~31% AU-PRO at default settings)

---

## 6. INP-Former (CVPR 2025)

**Paper**: "Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection"
**Code**: https://github.com/luow23/INP-Former
**Link**: https://arxiv.org/abs/2503.02424

### 6.1 Architecture

```
[Input Image]
    |
[Pre-trained Encoder] DINOv2-Reg ViT-Base/14 (frozen)
    | Multi-scale features from intermediate layers
    |
[INP Extractor] Cross-attention mechanism
    | M learnable query tokens (M=6 by default)
    | Query = learnable tokens, Key/Value = encoder features
    | Output: M Intrinsic Normal Prototypes (INPs)
    | Trained with INP Coherence Loss
    |
[Bottleneck] (feature compression)
    |
[INP-Guided Decoder] 8 Transformer layers
    | Modified attention: Query = patch tokens, Key/Value = INPs
    | Reconstructs features using ONLY normal prototypes
    | Anomalous regions cannot find matching INPs -> poor reconstruction
    |
[Segmentation Head]
    | Maps reconstruction error to anomaly score
    |
[Anomaly Score] = Reconstruction error (per-pixel)
```

### 6.2 How INPs Are Extracted from a Single Image

The key innovation: **normal prototypes are extracted FROM the test image itself**, not from an external memory bank.

**Mechanism**:
1. M learnable query tokens (M=6) attend to all patch tokens via cross-attention
2. Each INP becomes a weighted combination of the image's own tokens
3. INP Coherence Loss ensures these prototypes faithfully represent the image's normal patterns:
   - Minimizes distance between normal features and their nearest INP
   - Soft variant prevents all features collapsing to one INP
4. At inference: anomalous patches have no good INP to match -> high reconstruction error

**Why this works**: In most anomaly detection scenarios, anomalous regions are a small fraction of the image. The majority of patches are normal. So 6 prototypes extracted from the image are overwhelmingly dominated by normal patterns. The anomalous patches become "orphans" with no prototype to reconstruct from.

### 6.3 Loss Functions

**INP Coherence Loss**: Ensures INPs represent normality
- For each normal token, minimize distance to nearest INP
- Soft version: assign each token to a weighted combination of INPs (prevents degenerate solutions where all features map to one INP)

**Soft Mining Loss**: Prioritizes hard-to-optimize samples
- Focuses training on samples where reconstruction is difficult
- Prevents the model from ignoring challenging normal patterns

**Training**: StableAdamW optimizer, LR=1e-4, 200 epochs. Loss weights: lambda_1=3.0, lambda_2=0.2. No hyperparameter adjustment needed across MVTec/VisA/Real-IAD.

### 6.4 Results

| Dataset | Setting | I-AUROC | I-AP | I-F1max | P-AUROC | AUPRO |
|---------|---------|---------|------|---------|---------|-------|
| MVTec AD | Multi-class | 99.7 | 99.9 | 99.2 | 98.5 | 94.9 |
| VisA | Multi-class | ~98.5 | - | - | ~98.0 | - |
| Real-IAD | Multi-class | 90.5 | 88.1 | 81.5 | 99.0 | 95.0 |

INP-Former++ (extended version) achieves:
- MVTec AD multi-class: 99.8 I-AUROC, 96.0 AUPRO
- Semi-supervised: 99.7 I-AUROC, 96.8 AUPRO

### 6.5 Universal AD Capabilities

INP-Former is notable for performing well across:
- **Single-class AD**: Standard per-category training
- **Multi-class AD**: Single model for all categories
- **Few-shot AD**: Limited training samples
- **Zero-shot AD**: Some capability demonstrated (no training data for target)
- **Semi-supervised AD** (INP-Former++ only)

### 6.6 Strengths and Limitations

**Strengths**:
1. No external memory bank needed — prototypes come from the test image
2. Naturally handles multi-class because INPs adapt per image
3. Same hyperparameters work across datasets (no tuning needed)
4. Strong AUPRO (localization) performance on Real-IAD

**Limitations**:
1. **Assumption: anomalies are minority**: If anomalous regions cover a large portion of the image, INPs may capture anomalous patterns too, collapsing the reconstruction error signal.
2. **Fixed M=6 prototypes**: May not capture the diversity of normal patterns for very complex images. Too few for complex textures, too many for simple objects.
3. **Single-image prototype extraction**: No cross-image consistency. Two images of the same normal object may yield different INPs, making it harder to calibrate anomaly thresholds.
4. **GT mask accuracy note**: Authors acknowledge pixel-level AP and F1-max from code may be slightly lower than paper-reported numbers.
5. **Computational overhead**: Cross-attention for INP extraction + INP-guided decoding adds latency compared to simple reconstruction methods.
6. **No explicit handling of distribution shift**: Like Dinomaly, no evaluation on MVTec AD 2 lighting scenarios.

---

## Cross-Paper Comparison Summary

| Aspect | Dinomaly | AnomalyCLIP | PatchCore | ResAD | EfficientAD | INP-Former |
|--------|----------|-------------|-----------|-------|-------------|------------|
| **Type** | Reconstruction | VLM zero-shot | Memory bank | Residual NF | Student-Teacher | Reconstruction |
| **Backbone** | DINOv2-Reg ViT | CLIP ViT-L/14 | WideResNet-101 | WideResNet-50 | PDN (4-layer CNN) | DINOv2-Reg ViT |
| **Training** | Per-dataset | Prompt learning | No training | Per-class or unified | Per-class | Per-dataset |
| **Multi-class** | Yes (strong) | N/A (zero-shot) | No (per-class bank) | Yes (class-generalizable) | No (per-class) | Yes (strong) |
| **Zero-shot** | No | Yes | No | Partial (few-shot) | No | Partial |
| **Speed** | Moderate | Slow | Slow (NN search) | Moderate | Very fast (<2ms) | Moderate |
| **MVTec I-AUROC** | 99.6% | ~91.5% (0-shot) | 99.1% | ~90.5% (cross) | 99.8% | 99.7% |
| **Dist. shift robust** | Unknown | Unknown | Yes (~3pp drop) | Unknown | No (>10pp drop) | Unknown |
| **Logical anomalies** | No | Partial | No | No | Yes (AE module) | No |
| **Memory** | Low | Low | High (bank) | Moderate (references) | Low | Low |

## Key Research Gaps Identified

1. **Distribution shift robustness**: Only PatchCore/RD evaluated on MVTec AD 2. Most methods are untested. This is a wide-open direction.
2. **Multi-class + robustness**: No method achieves both strong multi-class performance AND robustness to distribution shift.
3. **Foundation model exploitation**: Dinomaly and INP-Former use DINOv2 but in different ways. Neither exploits the full potential of foundation model features for robustness.
4. **Residual features + reconstruction**: ResAD's residual insight hasn't been combined with reconstruction-based methods (Dinomaly/INP-Former). This could be a novel combination.
5. **Prototype-based robustness**: INP-Former's per-image prototypes could potentially provide natural robustness to distribution shift (prototypes adapt to current conditions), but this hasn't been explored.
6. **Logical anomaly detection in multi-class**: EfficientAD's AE approach is the only one addressing logical anomalies, but it's per-class and fragile. No multi-class logical anomaly method exists.

## 관련 노트
- analysis: `skill_graph/analysis/vad_trend/`
- papers: `skill_graph/papers/2026-03-23_핵심논문_목록.md`
- papers: `skill_graph/papers/2026-03-23_FM_AD_robustness_핵심논문.md`
