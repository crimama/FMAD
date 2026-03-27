# Feature Disentanglement for Foundation Model-based Anomaly Detection: Deep Research Survey

**Date**: 2026-03-23
**Purpose**: #1 novelty gap 탐색 — FM-AD에서 feature disentanglement 적용 가능성 조사
**Status**: Research survey (no code)

---

## Table of Contents
1. [Style-Content Disentanglement Theory](#1-style-content-disentanglement-theory)
2. [StyLIP and CLIP Feature Disentanglement](#2-stylip-and-clip-feature-disentanglement)
3. [DINOv2 Feature Space Structure](#3-dinov2-feature-space-structure)
4. [Feature Disentanglement for Anomaly Detection](#4-feature-disentanglement-for-anomaly-detection)
5. [Potential Approaches for FM-AD](#5-potential-approaches-for-fm-ad-feature-disentanglement)
6. [Synthesis and Research Direction](#6-synthesis-and-research-direction)

---

## 1. Style-Content Disentanglement Theory

### 1.1 von Kugelgen et al. (NeurIPS 2021) — Foundational Identifiability Theory

**Paper**: "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style"
**Authors**: von Kugelgen, Sharma, Gresele, Brendel, Scholkopf, Besserve, Locatello
**Venue**: NeurIPS 2021

#### Core Setup
- Data generating process: latent variable `z = (z_c, z_s)` where `z_c` = content, `z_s` = style
- Observation `x = g(z_c, z_s)` through mixing function `g`
- Augmentation preserves content `z_c` but changes style `z_s`
- **Key relaxation**: allows nontrivial statistical AND causal dependencies between `z_c` and `z_s` (unlike prior ICA work requiring independence)

#### Identifiability Theorems
- **Block identifiability**: content partition `z_c` identifiable up to an invertible mapping (not element-wise, but as a block)
- Proved in both **generative** (VAE-like) and **discriminative** (contrastive) settings
- **Augmentation defines the partition**: whatever is invariant across augmented views = content; whatever changes = style
- Content can be learned **without knowing the exact latent dimension** — practical advantage

#### Key Implications for AD
- Self-supervised features (DINO, CLIP) are trained with augmentations that define an implicit content/style partition
- **Standard augmentations** (crop, color jitter, blur) define "content" as spatial/semantic structure and "style" as color/texture/scale
- This means DINOv2 features *theoretically* should have content (semantic identity) isolated from style (appearance variations)
- **BUT**: the content/style boundary depends entirely on augmentation choice — anomaly-relevant info (texture defects, structural damage) might land in either partition depending on the pretraining augmentation strategy

#### Limitations
- Block identifiability only — cannot guarantee element-wise disentanglement
- Requires augmentations to be "rich enough" to span style variations
- No guarantee about what the identified blocks correspond to semantically
- Practical SSL may not perfectly satisfy the theoretical conditions

### 1.2 Zimmermann et al. (ICML 2021) — Contrastive Learning Inverts the DGP

**Paper**: "Contrastive Learning Inverts the Data Generating Process"

#### Core Contribution
- Proves feedforward models trained with InfoNCE learn to implicitly invert the underlying generative model
- Connection: contrastive learning ↔ nonlinear ICA ↔ generative model inversion

#### Relevance to Feature Entanglement
- Invariance: positive pairs (augmentations) → features insensitive to augmentation-defined "style"
- Sufficiency: negative pairs → features preserve enough information for downstream discrimination
- **Tension**: invariance pushes out style info, sufficiency pulls in discriminative info
- For AD: features may discard texture/appearance variations that are actually anomaly-relevant

### 1.3 Theoretical Landscape Summary

| Property | Guarantee | Practical Status |
|----------|-----------|-----------------|
| Content isolation | Block-identifiable up to invertible map | Strong for semantic content |
| Style isolation | Block-identifiable under sufficient conditions | Augmentation-dependent |
| Element-wise disentanglement | NOT guaranteed | Requires additional constraints |
| Anomaly-relevant separation | NOT addressed | **Open problem** — our opportunity |

---

## 2. StyLIP and CLIP Feature Disentanglement

### 2.1 StyLIP (WACV 2024)

**Paper**: "StyLIP: Multi-Scale Style-Conditioned Prompt Learning for CLIP-based Domain Generalization"
**Authors**: Bose et al.

#### Core Method
- Goal: disentangle visual style and content in CLIP's vision encoder for domain generalization
- **Style projectors**: learn domain-specific prompt tokens from extracted multi-scale style features
- Style features extracted from CLIP ViT at multiple scales → projected to prompt token space
- Content features combined with style-conditioned prompts for classification

#### Technical Architecture
- Extract intermediate features from CLIP ViT at multiple layer depths
- Style projectors (learnable MLPs) map style features → domain-specific prompt tokens
- Combines style prompts with content-aware classification prompts
- Domain-agnostic prompt learning: no explicit domain labels needed

#### Key Results
- Outperforms DG baselines by 0.2-4% across multiple benchmarks
- Shows CLIP features DO contain disentanglable style/content information
- Multi-scale extraction important — single-scale insufficient

#### Connection to AD
- **Validates**: CLIP features encode separable style/content at multiple layers
- **Gap**: StyLIP targets domain generalization (classify correctly across domains), not anomaly detection
- **Opportunity**: analogous multi-scale disentanglement could separate anomaly-relevant from domain-specific features

### 2.2 PISCO (ICML 2023)

**Paper**: "Simple Disentanglement of Style and Content in Visual Representations"
**Authors**: Ngweta, Maity, Gittens, Sun, Yurochkin (IBM Research)

#### Core Method
- **Post-hoc** disentanglement: works on frozen pretrained features (no retraining!)
- Models pretrained features probabilistically as **linearly entangled combinations** of latent content and style
- Assumption: `f(x) = A_c * z_c + A_s * z_s` (linear mixing)
- Simple algorithm to recover `z_c` and `z_s` from the learned features

#### Theoretical Guarantee
- Provably disentangles content and style under the linear mixing assumption
- Much stronger than most empirical disentanglement methods

#### Key Results
- Significant DG improvements when shift is due to image corruption or style-related spurious correlations
- Works with standard pretrained models (ResNet, ViT)
- Scales to ImageNet-scale datasets (unlike many disentanglement methods)

#### Connection to AD — **HIGH RELEVANCE**
- **Post-hoc + frozen features**: directly applicable to FM-AD pipelines (PatchCore, etc.)
- If foundation model features are linearly entangled mixtures of content and style, PISCO-style decomposition could:
  - Separate semantic structure (content) from appearance variations (style)
  - Use style-filtered features for AD → robust to domain shift
  - Use content-filtered features for AD → robust to texture anomalies that manifest in style
- **Key question**: is the linear mixing assumption valid for DINOv2/CLIP features?

---

## 3. DINOv2 Feature Space Structure

### 3.1 Artifact Problem — SINDER (ECCV 2024 Oral)

**Paper**: "SINDER: Repairing the Singular Defects of DINOv2"
**Authors**: Wang, Zhang, Salzmann (EPFL)

#### The Defect
- ~2.37% of DINOv2 patch tokens have abnormally high norms (~434 vs ~57 normal)
- These artifacts appear primarily in **middle-to-late layers** (after layer 20)
- Artifacts share high similarity across the ENTIRE dataset — not image-specific

#### Root Cause: Leading Left Singular Vector
- Linearizing transformer blocks as matrices `G_i = E_i...E_0`
- Defect direction strongly correlates with leading left singular vector of composed operations
- **Power method analogy**: repeated multiplication by the layer matrix amplifies the leading eigenvector
- When largest eigenvalue > 1, norm explodes along this direction
- After layer 20, angle between theoretical and empirical defect directions < 40 degrees

#### Fix: Smooth Regularization
- Identify defective tokens (>4σ from mean logit)
- Learning targets: weighted 3x3 spatial neighbors
- L2 loss to smooth defective tokens toward neighbors
- Freeze U, V matrices; learn only singular values → minimal parameters
- **6 hours on single V100 with 30k images** (vs full retraining with 142M images)

#### Results
- +4.2% mIoU Cityscapes, +7.1% VOC2012 (unsupervised segmentation)
- +2.28% mIoU ADE20k (supervised)
- Minimal classification degradation (-0.02%)

#### Connection to AD
- High-norm artifacts could create false anomaly signals or mask real anomalies
- Artifact tokens contain global image info but discard spatial info → problematic for localization-based AD
- **SINDER repair as preprocessing for FM-AD**: could improve feature quality before anomaly scoring

### 3.2 Register Tokens (ICLR 2024)

**Paper**: "Vision Transformers Need Registers"

#### Key Findings
- High-norm outlier tokens appear in redundant patches during inference
- These tokens **recycle** spatial positions to store global information
- Contain strong class-level information but poor positional/local info
- Adding learnable "register" tokens during pretraining eliminates artifacts entirely

#### Implication for AD
- DINOv2-reg (with registers) produces smoother, more consistent feature maps
- For patch-level AD: register-equipped models may give more reliable local features
- **Trade-off**: registers absorb global info → patch tokens become more local/textural

### 3.3 Phi-eat (arXiv 2511.11270, 2025)

**Paper**: "Phi-eat: Physically-Grounded Feature Representation"

#### Core Problem Identified
- DINOv2/DINOv3 features **entangle high-level semantics with low-level physical factors** (geometry, illumination, reflectance)
- Standard SSL pretraining optimizes for semantic consistency → suppresses material/texture cues

#### Method: Physical Augmentation Pretraining
- Replace photometric augmentations with **physical augmentations**: same material rendered on different geometries under varying illumination
- Contrastive learning with physical variations → features encode material identity
- DINOv3-based teacher-student framework, initialized from DINOv3

#### Quantitative Evidence of Entanglement
| Metric | DINOv2 | DINOv3 | Phi-eat |
|--------|--------|--------|---------|
| Material IoU | 0.566 | 0.599 | **0.776** |
| k-NN Top-1 | 0.563 | 0.600 | **0.643** |

- DINOv2 clusters features by **object parts** (semantic), mixing different materials
- Phi-eat clusters by **material properties** (physical), grouping same-material regions across objects

#### Critical Insight for AD
- **DINOv2 features conflate "what the object is" with "what it's made of / what it looks like"**
- For AD: a scratch on metal and a normal edge might have similar semantic features but different material features
- Phi-eat demonstrates that the entanglement is real and fixable with different pretraining
- **But**: retraining from scratch is impractical for us → need post-hoc methods

### 3.4 AD-DINOv3 — CLS Token Bias

**Paper**: "AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration"

#### CLS Token Bias Problem
- CLS token is "heavily biased toward generic foreground objects" due to natural-image pretraining
- Model confuses anomalies with salient object regions
- Subtle defects misinterpreted as part of normal foreground

#### Multi-Layer Feature Analysis
- Layers 6, 12, 18, 24 contain complementary information
- Early layers: low-level appearance cues
- Late layers: high-level semantic context
- Multi-level fusion: +1.22% AUROC over single-layer

#### Feature Space Observation
- "DINOv3 representations fragment normal regions into multiple clusters due to intra-class variations"
- "Anomalies fail to form a distinctive cluster"
- This fragmentation = evidence of entangled style/content in the AD context

### 3.5 DINOv2 Layer Hierarchy Summary

| Layer Range | Encodes | AD Relevance |
|-------------|---------|--------------|
| Early (1-6) | Low-level: edges, colors, textures | Texture anomalies, surface defects |
| Middle (7-18) | Mid-level: parts, patterns, structures | Structural anomalies |
| Late (19-24+) | High-level: object semantics, category | Object identity (confounding for AD) |
| CLS token | Global foreground semantics | Biased — misses subtle anomalies |

**Key problem**: AD needs mid/low-level features (where defects manifest) but these are entangled with high-level semantic variations (object identity, viewpoint, background).

---

## 4. Feature Disentanglement for Anomaly Detection

### 4.1 HOOD (ICLR 2023) — Causal Content/Style for OOD Detection

**Paper**: "Harnessing Out-Of-Distribution Examples via Augmenting Content and Style"
**Authors**: Huang, Xia, Shen, Han, Gong, Gong, Liu

#### Method
- Structural Causal Model (SCM) for image generation with content and style as latent causes
- Variational inference to disentangle content and style from observations
- **Intervention-based augmentation**:
  - Intervene on style → generate benign OOD (novel style, known content)
  - Intervene on content → generate malign OOD (unknown content, familiar style)

#### Key Insight for AD
- **Benign OOD** (style shift): same object, different appearance → should be normal in AD
- **Malign OOD** (content shift): different structure, same appearance → should be anomalous in AD
- This maps directly to the AD robustness problem: we want detectors invariant to style but sensitive to content anomalies
- **Limitation**: requires OOD examples for training, not directly applicable to one-class AD

### 4.2 FP-CLIP (2025) — Foreground-Background Separation

**Paper**: "FP-CLIP: Foreground-Panorama Prompt Learning for Zero-Shot Anomaly Detection"

#### Problem
- High false positive rate from confusing background anomalies with foreground defects
- CLIP features entangle foreground object with background context

#### Method
- LLM (Llama3.2-vision) generates foreground and panorama captions
- Four learnable prompts: foreground, panorama, normal, abnormal
- V-V Attention mechanism fuses global and local features
- Contrastive alignment between text features and image features

#### Connection to Disentanglement
- Separates foreground (object-of-interest) from background (context/panorama)
- This is spatial disentanglement rather than feature-level disentanglement
- Reduces false positives from background clutter
- Does NOT address within-object style/content separation

### 4.3 IB-IUMAD (2026) — Inter-Object Feature Coupling

**Paper**: "Towards an Incremental Unified Multimodal Anomaly Detection"

#### Problem
- In unified (multi-class) AD, features from different object categories interfere
- "Spurious feature interference between objects" during reconstruction

#### Method
- Mamba decoder to disentangle inter-object feature coupling
- Information Bottleneck Fusion Module (IBFM) filters redundant cross-modal features
- Addresses catastrophic forgetting in incremental AD setting

#### Connection to Disentanglement
- Addresses **inter-class** feature coupling, not **intra-feature** style/content separation
- Different problem from what we're targeting, but validates that feature coupling is a recognized problem in AD

### 4.4 FiCo (AAAI 2025) — Filter or Compensate for Distribution Shift in AD

**Paper**: "Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection"

#### Core Method
- **Distribution-Specific Compensation (DiSCo)**: reconstructs domain-specific info using dynamic convolution + instance normalization
- **Distribution-Invariant Filter (DiIFi)**: filters out distribution-specific info, retaining anomaly-relevant invariant features
- Built on Reverse Distillation framework

#### Key Results
- MVTec: 98.8% in-distribution, 97.6% average across OOD corruptions
- 1.22% improvement over prior SOTA (GNL)

#### Connection to Disentanglement
- Implicitly separates distribution-invariant (anomaly-relevant) from distribution-specific (style) features
- No explicit disentanglement loss — uses consistency regularization across augmented views
- No theoretical guarantees of disentanglement
- **Architecture-specific** (reverse distillation) — not generalizable to other FM-AD methods

### 4.5 AnomalyCLIP (ICLR 2024) — Object-Agnostic Prompts

**Paper**: "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection"

#### Core Insight
- VLMs focus on **class semantics of foreground objects** rather than abnormality/normality
- Need to disentangle "what the object is" from "is it normal/abnormal"

#### Method
- Learn object-agnostic text prompts capturing generic normality/abnormality
- Image-level + pixel-level loss for global and local anomaly patterns
- Works across 17 diverse datasets without retraining

#### Connection to Disentanglement
- Addresses the semantic-vs-anomaly entanglement in CLIP text space
- Prompt-level disentanglement (text side), not feature-level (vision side)
- Complementary to our proposed direction

### 4.6 ResAD (NeurIPS 2024 Spotlight) — Residual Feature Learning

**Paper**: "ResAD: A Simple Framework for Class Generalizable Anomaly Detection"

#### Core Insight
- **Residual features** = input feature - nearest normal reference feature
- Subtracting the nearest normal neighbor eliminates most class-related variation
- Residual features cluster around origin → class-agnostic anomaly detection

#### Method
1. Feature Converter: input → residual (subtract nearest normal reference)
2. Feature Constraintor: compress normal residuals into hypersphere (OCC-inspired)
3. Feature Distribution Estimator: model normal residual distribution

#### Connection to Disentanglement
- **Implicitly disentangles**: class identity (removed by subtraction) from anomaly signal (residual)
- Simple, effective, but loses spatial structure in the subtraction
- Does not address within-class style variation (lighting, pose, etc.)

### 4.7 SubspaceAD (2025) — PCA-based Subspace Modeling

**Method**: Fit PCA to DINOv2 patch features from few normal images; detect anomalies via reconstruction residual

#### Connection to Disentanglement
- PCA principal components = directions of maximum normal variance (dominated by style/class)
- **Orthogonal complement** = directions of minimal normal variance (anomaly-sensitive)
- Low-variance components encode variations suppressed in normal data but exaggerated in anomalies
- This IS a form of implicit disentanglement: principal subspace ≈ normal variation, null space ≈ anomaly signal

### 4.8 CausalCLIP (arXiv 2512.13285, 2025) — Causal Feature Filtering

**Paper**: "CausalCLIP: Causally-Informed Feature Disentanglement and Filtering"

#### Method
- Structural causal model for feature generation
- Factorization Module: splits CLIP features into causal and non-causal components via correlation analysis
- Gumbel-Softmax masking + HSIC independence constraints for statistical independence
- Retains only transferable forensic cues

#### Connection to AD
- Different task (generated image detection), but method architecture is directly transferable
- Demonstrates that CLIP features CAN be factorized into causal/non-causal components
- HSIC-based independence enforcement could be adapted for anomaly-relevant/irrelevant separation

---

## 5. Potential Approaches for FM-AD Feature Disentanglement

### 5.1 The Core Problem

Foundation model features (DINOv2, CLIP) encode a mixture of:
- **Anomaly-relevant**: texture, local structure, material properties, surface condition
- **Anomaly-irrelevant**: object identity, viewpoint, lighting, background, scale

Current FM-AD methods use these entangled features directly → sensitivity to domain shift, false positives from normal variation, missed subtle anomalies masked by semantic features.

### 5.2 Key Technical Challenges Specific to AD

1. **One-class problem**: no anomaly examples to define what "anomaly-relevant" means
2. **Unknown anomaly types**: cannot specify which features will be relevant a priori
3. **Asymmetric information**: have many normals but feature variation in normals is both style AND content
4. **Post-hoc constraint**: retraining foundation models is impractical → need methods that work on frozen features
5. **Localization requirement**: need patch-level (not just image-level) disentanglement
6. **Computational budget**: method must be lightweight enough to not negate FM efficiency advantages

### 5.3 Approach Family A: Linear Post-hoc Decomposition (PISCO-inspired)

**Idea**: Apply linear disentanglement to frozen FM features

**How it could work**:
1. Extract DINOv2 patch features from normal training images
2. Model features as linear mixture: `f = W_c * z_c + W_s * z_s`
3. Use domain/augmentation labels or statistical properties to identify the mixing matrices
4. Project features onto content-only or style-only subspace for AD scoring

**Strengths**:
- Theoretically grounded (provable under linear assumption)
- Post-hoc, frozen features, lightweight
- PISCO shows this works for DG with pretrained models

**Challenges**:
- Linear assumption may not hold for DINOv2 features
- Need a way to define "style" for AD (unlike DG where domain labels exist)
- What if anomaly signal is nonlinearly entangled?

### 5.4 Approach Family B: Augmentation-Defined Disentanglement (von Kugelgen-inspired)

**Idea**: Use AD-specific augmentations to define the content/style boundary

**How it could work**:
1. Define "style augmentations" = variations that should NOT affect normality (lighting, color, minor pose)
2. Define "content" = whatever is invariant to these augmentations
3. Fine-tune a lightweight adapter on DINOv2 features with contrastive loss on augmented pairs
4. The adapted features isolate anomaly-relevant content from style nuisance

**Strengths**:
- Theoretically justified by von Kugelgen's identifiability results
- Augmentation choice = explicit control over what to disentangle
- Adapter-based = practical and lightweight

**Challenges**:
- Need to carefully design augmentations that preserve anomaly signals
- Risk: augmentations that also change anomaly appearance → anomaly info leaks into "style"
- Only guarantees block identifiability, not element-wise

### 5.5 Approach Family C: Subspace Decomposition with Anomaly Prior

**Idea**: Decompose FM feature space into normal-variation subspace and anomaly-sensitive complement

**How it could work**:
1. PCA on normal training features → principal subspace captures normal variations
2. **Key insight**: normal variation subspace ≈ style + class identity; complement ≈ anomaly-sensitive
3. Score anomalies based on projection onto complement subspace (residual-based)
4. **Enhancement**: use multi-layer features; different layers have different normal-variation structures
5. Weighted combination of per-layer residuals

**Strengths**:
- Training-free or minimal training
- SubspaceAD already validates this works
- Interpretable: can analyze what each subspace captures

**Challenges**:
- Assumes anomalies deviate from normal subspace (may miss in-subspace anomalies)
- PCA is linear — may miss nonlinear anomaly patterns
- Subspace dimension selection is critical and dataset-dependent

### 5.6 Approach Family D: Causal Feature Factorization (CausalCLIP/HOOD-inspired)

**Idea**: Build a causal model where structural features cause normality/anomaly, and nuisance factors are confounders

**How it could work**:
1. SCM: structural_feature → normal/anomaly ← nuisance_factor
2. Use variational inference to separate structural from nuisance factors
3. Gumbel-Softmax masking to select anomaly-causal feature dimensions
4. HSIC constraint to enforce independence between causal and non-causal features

**Strengths**:
- Principled causal framework
- Can handle confounders (lighting changes that correlate with anomaly appearance)
- CausalCLIP validates factorization module works on CLIP features

**Challenges**:
- Requires training signal → how to train without anomaly labels?
- SCM assumptions may be too strong
- Computational overhead of variational inference

### 5.7 Approach Family E: Multi-Layer Selective Fusion

**Idea**: Different DINOv2 layers encode different information; selectively fuse layers based on anomaly-relevance

**How it could work**:
1. Early layers (1-6): texture/surface features → anomaly-relevant for surface defects
2. Middle layers (7-18): structural features → anomaly-relevant for structural defects
3. Late layers (19-24): semantic features → mostly anomaly-irrelevant (class identity)
4. Learn attention weights over layers that emphasize anomaly-relevant layers
5. Suppress late-layer semantic features that cause false positives

**Strengths**:
- Leverages known DINOv2 layer hierarchy
- AD-DINOv3 already shows multi-layer helps
- Simple implementation

**Challenges**:
- Optimal layer selection may be category-dependent
- Not true disentanglement — just layer selection
- Semantic features in late layers may still be useful for some anomaly types

---

## 6. Synthesis and Research Direction

### 6.1 What Exists vs What's Missing

| Aspect | Exists | Missing |
|--------|--------|---------|
| Theory: content/style identifiability | von Kugelgen (NeurIPS 2021) | Application to AD setting |
| Linear post-hoc disentanglement | PISCO (ICML 2023) | Validation on FM-AD features |
| CLIP style/content separation | StyLIP (WACV 2024) | Applied to AD, not just DG |
| DINOv2 feature space analysis | SINDER, Phi-eat, AD-DINOv3 | Systematic anomaly-relevant vs irrelevant mapping |
| Causal disentanglement for OOD | HOOD (ICLR 2023), CausalCLIP | Adaptation to one-class AD setting |
| AD with distribution shift | FiCo (AAAI 2025) | Principled (not implicit) disentanglement |
| Feature residual for class-agnostic AD | ResAD (NeurIPS 2024) | Does not address within-class style variation |
| Subspace-based AD | SubspaceAD (2025) | No explicit disentanglement framework |

### 6.2 The Novelty Gap

**No existing work explicitly and principally disentangles FM features into anomaly-relevant vs anomaly-irrelevant components for AD.**

Current approaches either:
- Do implicit separation (FiCo, ResAD, SubspaceAD) without theoretical grounding
- Apply disentanglement to non-AD tasks (StyLIP for DG, PISCO for DG, HOOD for OOD classification)
- Analyze DINOv2 features (SINDER, Phi-eat) without connecting to AD disentanglement

### 6.3 Most Promising Direction

**Post-hoc Feature Disentanglement for FM-AD** combining:

1. **Theoretical foundation**: von Kugelgen's identifiability + PISCO's linear decomposition
2. **AD-specific formulation**: define "content" = structural normality signal, "style" = appearance nuisance
3. **Practical mechanism**: lightweight post-hoc method on frozen DINOv2/CLIP features
4. **Multi-layer awareness**: leverage known layer hierarchy (early=texture, late=semantic)
5. **One-class compatible**: derive the decomposition from normal data only

**Candidate paper message**: "Foundation model features entangle anomaly-relevant structural information with anomaly-irrelevant appearance variations. We propose [method] that provably disentangles these components, enabling robust anomaly detection under domain shift while preserving sensitivity to genuine defects."

### 6.4 Key Experiments to Validate

1. **Feature analysis**: PCA/CKA analysis of DINOv2 features under controlled style/content variations
2. **Linear decomposition test**: Does PISCO-style linear separation work on DINOv2 AD features?
3. **Augmentation-based test**: Can augmentation-defined contrastive learning separate anomaly signal?
4. **Robustness benchmark**: performance under lighting/pose/background shifts (RobustAD, PACS)
5. **Ablation**: which layers benefit most from disentanglement?

---

## Sources

### Theory
- [von Kugelgen et al. NeurIPS 2021 - SSL Identifiability](https://arxiv.org/abs/2106.04619)
- [Zimmermann et al. ICML 2021 - Contrastive Learning Inverts DGP](https://arxiv.org/abs/2102.08850)

### CLIP/Domain Generalization
- [StyLIP WACV 2024](https://arxiv.org/abs/2302.09251)
- [PISCO ICML 2023](https://proceedings.mlr.press/v202/ngweta23a.html)

### DINOv2 Feature Space
- [SINDER ECCV 2024](https://arxiv.org/abs/2407.16826)
- [Phi-eat arXiv 2025](https://arxiv.org/abs/2511.11270)
- [AD-DINOv3](https://arxiv.org/abs/2509.14084)
- [Vision Transformers Need Registers ICLR 2024](https://arxiv.org/abs/2309.16588)

### Anomaly Detection
- [HOOD ICLR 2023](https://arxiv.org/abs/2207.03162)
- [FP-CLIP 2025](https://www.sciencedirect.com/science/article/abs/pii/S1051200425008140)
- [IB-IUMAD 2026](https://arxiv.org/html/2603.02629)
- [FiCo AAAI 2025](https://arxiv.org/abs/2412.10115)
- [AnomalyCLIP ICLR 2024](https://arxiv.org/abs/2310.18961)
- [ResAD NeurIPS 2024](https://arxiv.org/abs/2410.20047)
- [SubspaceAD 2025](https://arxiv.org/html/2602.23013)
- [CausalCLIP 2025](https://arxiv.org/abs/2512.13285)

---

## 관련 노트
- `../fm_ad_robustness/2026-03-23_mechanism_analysis.md` — FM-AD 실패 메커니즘 분석
- `../fm_ad_robustness/2026-03-23_solution_survey.md` — 기존 해결책 서베이
