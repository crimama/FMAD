# Foundation Model-based Visual Anomaly Detection: Comprehensive Paper Survey (2024-2026)

> **Last updated**: 2026-03-30
> **Scope**: Papers using DINOv2, CLIP, or other vision FMs for visual/industrial anomaly detection
> **Focus**: 2024-2026 publications at top venues (CVPR, ICLR, NeurIPS, ECCV, WACV, AAAI) + strong arXiv

---

## Table of Contents
1. [Reconstruction-based (DINOv2 backbone)](#1-reconstruction-based-dinov2-backbone)
2. [Zero-shot CLIP-based](#2-zero-shot-clip-based)
3. [Few-shot FM-based](#3-few-shot-fm-based)
4. [Unified / Multi-class](#4-unified--multi-class)
5. [Training-free / Nearest-neighbor](#5-training-free--nearest-neighbor)
6. [Diffusion-based](#6-diffusion-based)
7. [Multi-modal / Hybrid](#7-multi-modal--hybrid)
8. [Summary Table](#8-summary-table)

---

## 1. Reconstruction-based (DINOv2 backbone)

### 1.1 Dinomaly (CVPR 2025)
- **Title**: Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection
- **Link**: https://arxiv.org/abs/2405.14325
- **Venue**: CVPR 2025
- **Backbone**: DINOv2 (ViT)
- **Method type**: Full-shot / Multi-class unsupervised
- **Core method**: Minimalistic reconstruction-based framework using pure Transformer architecture. Identifies four essential components: (1) Foundation Transformer features, (2) Noisy Bottleneck via existing Dropouts, (3) Linear Attention that naturally cannot focus, (4) Loose Reconstruction without layer-to-layer point-by-point constraints.
- **Key insight**: Complex designs are unnecessary -- simple orchestration of minimal components with foundation model features achieves SOTA. First multi-class UAD model competitive with single-class SOTAs.
- **Performance**: I-AUROC 99.6% (MVTec AD), 98.7% (VisA), 89.3% (Real-IAD) in multi-class setting
- **Limitation**: Requires full training set of normal images; multi-class gap still exists vs single-class for some metrics; reconstruction-based approach may struggle with subtle logical anomalies.

### 1.2 Dinomaly2 (arXiv 2025, under review)
- **Title**: One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection
- **Link**: https://arxiv.org/abs/2510.17611
- **Venue**: arXiv (Oct 2025)
- **Backbone**: DINOv2
- **Method type**: Unified (multi-class, few-shot, multi-modal)
- **Core method**: Extends Dinomaly's "less is more" philosophy to full-spectrum UAD. Adds contextual recentering and loosened optimization objectives. Bridges multi-class, single-class, few-shot, and multi-modal settings in one framework.
- **Key insight**: A single minimalistic reconstruction framework can handle 2D, multi-view, RGB-3D, RGB-IR modalities and various task settings without architectural changes.
- **Performance**: I-AUROC 99.9% (MVTec AD), 99.3% (VisA) in multi-class; 98.7% / 97.4% with only 8-shot
- **Limitation**: Very strong results may be hard to improve upon; still reconstruction-based paradigm; requires normal training data.

### 1.3 INP-Former (CVPR 2025)
- **Title**: Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection
- **Link**: https://arxiv.org/abs/2503.02424
- **Venue**: CVPR 2025
- **Backbone**: Foundation Transformer (DINOv2-style)
- **Method type**: Universal (single-class, multi-class, few-shot, zero-shot capable)
- **Core method**: Extracts Intrinsic Normal Prototypes (INPs) directly from the test image itself by linearly combining normal tokens. INP Coherence Loss ensures INPs faithfully represent normality. INP-Guided Decoder reconstructs only normal tokens; reconstruction errors serve as anomaly scores.
- **Key insight**: Normal prototypes can be discovered within a single test image without external normal references, avoiding the alignment problem between training references and test images.
- **Performance**: SOTA in single-class, multi-class, and few-shot settings on MVTec-AD, VisA, Real-IAD; demonstrates zero-shot capability.
- **Limitation**: Linear combination assumption for normal prototypes may not hold for all anomaly types; extended version (INP-Former++) needed for semi-supervised setting.

---

## 2. Zero-shot CLIP-based

### 2.1 AnomalyCLIP (ICLR 2024)
- **Title**: AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection
- **Link**: https://arxiv.org/abs/2310.18961
- **Venue**: ICLR 2024
- **Backbone**: CLIP (ViT-L/14@336)
- **Method type**: Zero-shot
- **Core method**: Learns object-agnostic text prompts that capture generic normality/abnormality regardless of foreground objects. Trains on auxiliary datasets, then transfers to unseen categories/domains at zero-shot.
- **Key insight**: By making prompts object-agnostic (not class-specific), the model focuses on abnormal regions rather than object semantics, enabling cross-domain generalization.
- **Performance**: Superior zero-shot performance across 17 real-world AD datasets (industrial + medical)
- **Limitation**: Requires auxiliary anomaly data for prompt training; CLIP's patch-level features are coarser than DINOv2's; struggles with fine-grained structural anomalies.

### 2.2 AA-CLIP (CVPR 2025)
- **Title**: AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP
- **Link**: https://arxiv.org/abs/2503.06661
- **Venue**: CVPR 2025
- **Backbone**: CLIP
- **Method type**: Zero-shot
- **Core method**: Two-stage approach: (1) creates anomaly-aware text anchors to differentiate normal/abnormal semantics, (2) aligns patch-level visual features with these anchors for precise localization. Uses residual adapters for controlled adaptation.
- **Key insight**: CLIP is inherently "anomaly-unaware" -- its features don't distinguish normal from abnormal well. Explicitly injecting anomaly awareness into both text and visual spaces while preserving generalization solves this.
- **Performance**: SOTA in zero-shot AD for industrial and medical applications
- **Limitation**: Still depends on CLIP's spatial resolution; two-stage training adds complexity; anomaly-aware anchors may not generalize to completely novel anomaly types.

### 2.3 AdaCLIP (ECCV 2024)
- **Title**: AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection
- **Link**: https://arxiv.org/abs/2407.15795
- **Venue**: ECCV 2024
- **Backbone**: CLIP
- **Method type**: Zero-shot
- **Core method**: Hybrid learnable prompts: static prompts (shared across images for preliminary adaptation) + dynamic prompts (generated per test image for instance-specific adaptation). Trained on auxiliary anomaly data.
- **Key insight**: Static prompts alone lack instance-level adaptability; dynamic prompts alone lack stable category-level knowledge. Combining both yields robust zero-shot detection.
- **Performance**: Outperforms other ZSAD methods across 14 datasets (industrial + medical)
- **Limitation**: Requires auxiliary annotated anomaly data; dynamic prompt generation adds inference cost.

### 2.4 VCP-CLIP (ECCV 2024)
- **Title**: VCP-CLIP: A Visual Context Prompting Model for Zero-Shot Anomaly Segmentation
- **Link**: https://link.springer.com/chapter/10.1007/978-3-031-72890-7_18
- **Venue**: ECCV 2024
- **Backbone**: CLIP
- **Method type**: Zero-shot segmentation
- **Core method**: Pre-VCP module embeds global visual information into text prompts (eliminates product-specific prompts). Post-VCP module adjusts text embeddings using fine-grained image features.
- **Key insight**: Visual context from the image itself should guide prompt generation, making the system truly product-agnostic without manual prompt engineering.
- **Performance**: Competitive zero-shot anomaly segmentation on MVTec AD and VisA
- **Limitation**: Still limited by CLIP's spatial resolution for fine-grained segmentation.

### 2.5 WinCLIP (CVPR 2023)
- **Title**: WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
- **Link**: https://arxiv.org/abs/2303.14814
- **Venue**: CVPR 2023
- **Backbone**: CLIP
- **Method type**: Zero-shot / Few-shot
- **Core method**: Window-based CLIP with compositional ensemble on state words and prompt templates. Efficient extraction and aggregation of window/patch/image-level features aligned with text.
- **Key insight**: Multi-scale windowed features from CLIP, combined with compositional text prompts, can achieve zero-shot AD without any training.
- **Performance**: Zero-shot: 91.8% / 85.1% I-AUROC/P-AUROC (MVTec); 1-shot (WinCLIP+): 93.1% / 95.2%
- **Limitation**: Foundational but surpassed by later methods; window-based approach is computationally expensive; coarse localization.

---

## 3. Few-shot FM-based

### 3.1 AnomalyDINO (WACV 2025)
- **Title**: AnomalyDINO: Boosting Patch-based Few-Shot Anomaly Detection with DINOv2
- **Link**: https://arxiv.org/abs/2405.14529
- **Venue**: WACV 2025
- **Backbone**: DINOv2
- **Method type**: Few-shot (1-shot to 16-shot), training-free
- **Core method**: Adapts DINOv2 for few-shot AD with a methodologically simple, training-free approach. No additional data needed for fine-tuning or meta-learning. Uses patch-level feature matching with DINOv2's rich representations.
- **Key insight**: DINOv2's self-supervised features are already rich enough for few-shot AD without any adaptation -- simplicity wins over complex meta-learning.
- **Performance**: 1-shot MVTec AD: 96.6% I-AUROC (from previous 93.1% SOTA)
- **Limitation**: Performance degrades in zero-shot; patch matching may miss global structural/logical anomalies.

### 3.2 FoundAD (ICLR 2026)
- **Title**: Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors
- **Link**: https://arxiv.org/abs/2510.01934
- **Venue**: ICLR 2026
- **Backbone**: DINOv2 / DINOv3 / multiple foundation encoders
- **Method type**: Few-shot
- **Core method**: Observes that anomaly amount correlates directly with embedding differences. Learns a nonlinear projection operator onto the natural image manifold. Anomalies are identified as out-of-distribution regions relative to this learned manifold.
- **Key insight**: Foundation encoder embeddings inherently encode anomaly information -- the key is finding the right projection to expose it, not building complex detection heads.
- **Performance**: 4-shot MVTec AD: outperforms PromptAD by 0.7% P-AUROC and 3.1% PRO; best across both image-level and pixel-level metrics. Substantially fewer parameters than prior methods.
- **Limitation**: Requires learning the projection (not training-free); performance tied to foundation encoder quality.

### 3.3 IIPAD (ICLR 2025)
- **Title**: One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning
- **Link**: https://openreview.net/forum?id=Zzs3JwknAY
- **Venue**: ICLR 2025
- **Backbone**: CLIP + BLIP-Diffusion
- **Method type**: Few-shot, one-for-all
- **Core method**: First one-for-all few-shot AD method. Learns a class-shared prompt generator (not fixed per-class prompts) that adaptively generates suitable prompts for each instance. Aligns prompts with visual space using general textual descriptions. Addresses memory bank mismatch in one-for-all paradigm.
- **Key insight**: Instead of learning fixed prompts per category, a shared prompt generator can adaptively handle all categories, solving the scalability problem of per-class prompt learning.
- **Performance**: Superior few-shot AD on MVTec and VisA under one-for-all paradigm
- **Limitation**: Relies on BLIP-Diffusion for instance-specific generation; complex multi-model pipeline.

### 3.4 PromptAD (CVPR 2024)
- **Title**: PromptAD: Learning Prompts with Only Normal Samples for Few-Shot Anomaly Detection
- **Link**: https://arxiv.org/abs/2404.05231
- **Venue**: CVPR 2024
- **Backbone**: CLIP
- **Method type**: Few-shot
- **Core method**: One-class prompt learning using semantic concatenation -- normal prompts are transposed into anomaly prompts by concatenating with anomaly suffixes. Only requires normal samples for training.
- **Key insight**: Anomaly prompts can be derived from normal prompts by suffix concatenation, avoiding the need for real anomaly samples.
- **Performance**: 1st place in 11/12 few-shot settings on MVTec and VisA (image + pixel level)
- **Limitation**: Per-class prompt learning limits scalability; requires some normal samples.

### 3.5 MetaUAS (NeurIPS 2024)
- **Title**: MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning
- **Link**: https://arxiv.org/abs/2505.09265
- **Venue**: NeurIPS 2024
- **Backbone**: Pure vision foundation model (no VLM)
- **Method type**: One-prompt universal segmentation
- **Core method**: Unifies anomaly segmentation into change segmentation. Trains on synthetic image pairs (object-level and local region changes) derived from existing datasets, independent of anomaly datasets. Soft feature alignment bridges paired-image change perception and single-image segmentation.
- **Key insight**: First pure-vision approach to universal anomaly segmentation without VLMs. Anomaly detection can be reframed as change detection between a normal prompt and test image.
- **Performance**: Outperforms zero-shot, few-shot, and even full-shot anomaly segmentation methods
- **Limitation**: Relies on quality of synthetic training data; change detection framing may miss anomalies that don't manifest as clear visual changes.

---

## 4. Unified / Multi-class

### 4.1 UniAD (NeurIPS 2022 Spotlight)
- **Title**: A Unified Model for Multi-class Anomaly Detection
- **Link**: https://arxiv.org/abs/2206.03687
- **Venue**: NeurIPS 2022
- **Backbone**: Pre-trained CNN (not FM per se, but foundational work)
- **Method type**: Multi-class unified
- **Core method**: Layer-wise query decoder to model multi-class distribution; neighbor masked attention to avoid information leak. Addresses "identical shortcut" in reconstruction networks.
- **Key insight**: Query embeddings in attention prevent the reconstruction shortcut where both normal and anomalous samples are well recovered.
- **Performance**: 96.5% I-AUROC, 96.8% P-AUROC on MVTec (15 classes unified)
- **Limitation**: Pre-FM era backbone; surpassed by Dinomaly; relatively complex architecture.

### 4.2 UniVAD (CVPR 2025)
- **Title**: UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection
- **Link**: https://arxiv.org/abs/2412.03342
- **Venue**: CVPR 2025
- **Backbone**: Vision foundation models (DINOv2-based)
- **Method type**: Few-shot, training-free, unified cross-domain
- **Core method**: Three modules: (1) Contextual Component Clustering (C3) for component segmentation using vision FMs, (2) Component-Aware Patch Matching (CAPM) for local anomaly detection, (3) Graph-Enhanced Component Modeling (GECM) for structural/logical anomaly detection. Aggregates multi-level semantic results.
- **Key insight**: First training-free unified method across industrial, logical, AND medical domains. Component-level modeling bridges structural and logical anomaly detection.
- **Performance**: SOTA few-shot AD across 9 datasets spanning industrial, logical, and medical domains; outperforms domain-specific models
- **Limitation**: Component clustering quality depends on foundation model; graph modeling adds complexity; may struggle with very fine-grained defects.

---

## 5. Training-free / Nearest-neighbor

### 5.1 MuSc (ICLR 2024)
- **Title**: MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images
- **Link**: https://arxiv.org/abs/2401.16753
- **Venue**: ICLR 2024
- **Backbone**: DINOv2 / CLIP (feature extraction only)
- **Method type**: Zero-shot (no training, no prompts)
- **Core method**: Normal patches have many similar patches across unlabeled images; anomalous patches have few. Uses Local Neighborhood Aggregation with Multiple Degrees (LNAMD) for multi-scale patch features and Mutual Scoring Mechanism (MSM) for inter-image anomaly scoring.
- **Key insight**: Anomaly detection can be done purely by mutual comparison among unlabeled test images -- normal patterns are statistically common, anomalies are rare.
- **Performance**: +21.1% PRO absolute gain on MVTec AD (72.7% -> 93.8%); +19.4% pixel-AP on VisA
- **Limitation**: Requires a batch of test images (not single-image); performance depends on anomaly ratio in test batch; computationally expensive mutual comparison.

### 5.2 SubspaceAD (arXiv Feb 2026)
- **Title**: SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling
- **Link**: https://arxiv.org/abs/2602.23013
- **Venue**: arXiv (Feb 2026)
- **Backbone**: DINOv2 (frozen)
- **Method type**: Few-shot, training-free
- **Core method**: Two stages: (1) Extract patch-level features from few normal images via frozen DINOv2, (2) Fit PCA model to estimate low-dimensional normal subspace. Anomalies detected via reconstruction residual w.r.t. subspace.
- **Key insight**: PCA on frozen DINOv2 features is sufficient for SOTA few-shot AD -- the normal variation lies in a low-dimensional subspace, and anomalies project poorly onto it.
- **Performance**: SOTA few-shot AD with extremely simple method (PCA only)
- **Limitation**: PCA assumes linear subspace; may fail for complex non-linear normal variations; no learned adaptation.

### 5.3 SuperAD (CVPR 2025 VAND Workshop)
- **Title**: SuperAD: A Training-free Anomaly Classification and Segmentation Method
- **Link**: https://arxiv.org/abs/2505.19750
- **Venue**: CVPR 2025 VAND 3.0 Workshop (Challenge Track 1 winner)
- **Backbone**: DINOv2
- **Method type**: Training-free, adapt & detect
- **Core method**: Fully training-free AD based on DINOv2 feature extraction. Designed for the VAND 3.0 "Adapt & Detect" challenge, demonstrating superior generalization across unseen domains.
- **Key insight**: DINOv2 features alone, with appropriate scoring, can outperform trained methods in domain adaptation scenarios.
- **Performance**: Consistently outperforms previous methods on VAND 3.0 test sets
- **Limitation**: Workshop paper with less rigorous evaluation; specific to challenge setting.

---

## 6. Diffusion-based

### 6.1 One-for-More (CVPR 2025)
- **Title**: One-for-More: Continual Diffusion Model for Anomaly Detection
- **Link**: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection_CVPR_2025_paper.pdf
- **Venue**: CVPR 2025
- **Backbone**: Diffusion model
- **Method type**: Continual multi-class
- **Core method**: Continual learning approach for diffusion-based AD that incrementally learns new categories without forgetting previous ones. Single diffusion model handles growing set of product categories.
- **Key insight**: Diffusion models can be continually adapted for new AD categories, solving the practical problem of deploying AD systems that must handle new products over time.
- **Performance**: Competitive multi-class AD with continual learning capability
- **Limitation**: Diffusion inference is slow; continual learning adds training overhead; may not match FM-based methods on pure performance.

### 6.2 AnomalyXFusion (arXiv 2024)
- **Title**: AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion
- **Link**: https://arxiv.org/abs/2404.19444
- **Venue**: arXiv (May 2024)
- **Backbone**: Diffusion model + multi-modal inputs
- **Method type**: Anomaly synthesis (data augmentation)
- **Core method**: Multi-modal In-Fusion (MIF) combines image, text, and mask into "X-embedding". Dynamic Dif-Fusion (DDF) adaptively conditions generation on current diffusion step. Generates realistic anomaly samples for training.
- **Key insight**: Multi-modal conditioning (image + text + mask) produces far more realistic and diverse anomaly synthesis than image-only approaches, especially for logical anomalies.
- **Performance**: IS 1.82, classification accuracy 74.7% on MVTec AD (vs baseline max 66.1%)
- **Limitation**: Synthesis quality still imperfect; requires MVTec Caption annotations; generated anomalies may not cover all real defect modes.

---

## 7. Multi-modal / Hybrid

### 7.1 LogSAD (CVPR 2025)
- **Title**: Towards Training-free Anomaly Detection with Vision and Language Foundation Models
- **Link**: https://arxiv.org/abs/2503.18325
- **Venue**: CVPR 2025
- **Backbone**: CLIP + DINOv2 + GPT-4V (multi-modal)
- **Method type**: Training-free, logical + structural
- **Core method**: Match-of-thought architecture using GPT-4V to generate matching proposals. Multi-granularity detection: patch tokens + sets of interests + composition matching. Calibration module aligns scores from different detectors.
- **Key insight**: Large multi-modal models (GPT-4V) can generate compositional rules for logical anomaly detection, complementing patch-level structural detection.
- **Performance**: SOTA training-free AD, competitive even with supervised approaches
- **Limitation**: Requires GPT-4V API (cost, latency, reproducibility); complex multi-model pipeline; logical anomaly detection still challenging.

### 7.2 CLIP-DINOv2 Fusion (MDPI Electronics 2025)
- **Title**: Zero-Shot Industrial Anomaly Detection via CLIP-DINOv2 Multimodal Fusion and Stabilized Attention Pooling
- **Link**: https://www.mdpi.com/2079-9292/14/24/4785
- **Venue**: Electronics (Dec 2025)
- **Backbone**: CLIP + DINOv2
- **Method type**: Zero-shot
- **Core method**: Hierarchically aligns CLIP's global semantic embeddings with DINOv2's multi-scale structural features via Dual-Modality Attention. Stabilized Attention-based Pooling adaptively aggregates discriminative representations.
- **Key insight**: CLIP provides semantic (macro) understanding while DINOv2 provides structural (micro) understanding -- fusing both captures anomalies at all scales.
- **Performance**: 93.4% I-AUROC across 7 benchmarks (zero-shot)
- **Limitation**: Multi-backbone inference cost; fusion mechanism adds complexity; journal paper with potentially less rigorous review.

### 7.3 MVFA-AD (CVPR 2024 Highlight)
- **Title**: Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images
- **Link**: https://arxiv.org/abs/2403.12570
- **Venue**: CVPR 2024 (Highlight)
- **Backbone**: CLIP
- **Method type**: Generalizable (medical domain)
- **Core method**: Lightweight multi-level adaptation framework with multiple residual adapters inserted into CLIP's visual encoder. Stepwise enhancement of visual features across levels for medical AD.
- **Key insight**: Residual adapters at multiple levels can adapt CLIP for medical AD while preserving generalization -- minimal parameter overhead.
- **Performance**: Strong generalization across medical AD benchmarks
- **Limitation**: Medical-domain specific; residual adapters still require training; limited evaluation on industrial AD.

### 7.4 SAA+ (Segment Any Anomaly)
- **Title**: Segment Any Anomaly without Training via Hybrid Prompt Regularization
- **Link**: https://github.com/caoyunkang/Segment-Any-Anomaly
- **Venue**: CVPR 2023 VAND Workshop (2nd place)
- **Backbone**: CLIP + SAM + Grounding DINO
- **Method type**: Zero-shot segmentation
- **Core method**: Assembles multiple foundation models (SAM, CLIP, Grounding DINO) with multi-modal prompt regularization from domain expert knowledge and target image context.
- **Key insight**: Cascading foundation models with prompt-based regularization can achieve zero-shot anomaly segmentation without any training.
- **Performance**: SOTA zero-shot anomaly segmentation on VisA and MVTec AD
- **Limitation**: Complex multi-model pipeline; high inference cost; depends on all models working well together.

---

## 8. Summary Table

| # | Paper | Venue | Backbone | Type | MVTec I-AUROC | Key Contribution |
|---|-------|-------|----------|------|---------------|-----------------|
| 1 | Dinomaly | CVPR 2025 | DINOv2 | Full-shot MC | 99.6% | Minimalist reconstruction, first MC=SC |
| 2 | Dinomaly2 | arXiv 2025 | DINOv2 | Unified | 99.9% | Full-spectrum UAD |
| 3 | INP-Former | CVPR 2025 | DINOv2-style | Universal | SOTA | Intrinsic normal prototypes from test image |
| 4 | AnomalyCLIP | ICLR 2024 | CLIP | Zero-shot | ~91% | Object-agnostic prompt learning |
| 5 | AA-CLIP | CVPR 2025 | CLIP | Zero-shot | SOTA ZS | Anomaly-aware CLIP adaptation |
| 6 | AdaCLIP | ECCV 2024 | CLIP | Zero-shot | - | Static + dynamic hybrid prompts |
| 7 | VCP-CLIP | ECCV 2024 | CLIP | Zero-shot seg | - | Visual context prompting |
| 8 | WinCLIP | CVPR 2023 | CLIP | Zero/Few-shot | 91.8% (ZS) | Window-based multi-scale CLIP |
| 9 | AnomalyDINO | WACV 2025 | DINOv2 | Few-shot TF | 96.6% (1-shot) | Training-free DINOv2 few-shot |
| 10 | FoundAD | ICLR 2026 | DINOv2/v3 | Few-shot | SOTA FS | Manifold projection from FM embeddings |
| 11 | IIPAD | ICLR 2025 | CLIP+BLIP | Few-shot 1FA | SOTA 1FA | Instance-induced shared prompt generator |
| 12 | PromptAD | CVPR 2024 | CLIP | Few-shot | 1st/12 | Normal-to-anomaly prompt concatenation |
| 13 | MetaUAS | NeurIPS 2024 | Vision FM | 1-prompt | >full-shot | Change segmentation paradigm, pure vision |
| 14 | UniAD | NeurIPS 2022 | CNN | Multi-class | 96.5% | Query decoder, neighbor masked attention |
| 15 | UniVAD | CVPR 2025 | DINOv2 | Few-shot TF unified | SOTA cross-domain | First TF cross-domain unified AD |
| 16 | MuSc | ICLR 2024 | DINOv2/CLIP | Zero-shot | 93.8% PRO | Mutual scoring among unlabeled images |
| 17 | SubspaceAD | arXiv 2026 | DINOv2 | Few-shot TF | SOTA FS | PCA on frozen DINOv2 features |
| 18 | SuperAD | CVPR25 WS | DINOv2 | TF | Challenge win | VAND 3.0 challenge winner |
| 19 | One-for-More | CVPR 2025 | Diffusion | Continual MC | Competitive | Continual diffusion for AD |
| 20 | AnomalyXFusion | arXiv 2024 | Diffusion | Synthesis | 74.7% cls | Multi-modal anomaly synthesis |
| 21 | LogSAD | CVPR 2025 | CLIP+DINOv2+GPT4V | TF logical+struct | SOTA TF | Multi-modal match-of-thought |
| 22 | CLIP-DINOv2 Fusion | Electronics 2025 | CLIP+DINOv2 | Zero-shot | 93.4% | Dual-modality attention fusion |
| 23 | MVFA-AD | CVPR 2024 HL | CLIP | Medical AD | - | Multi-level residual adapters for medical |
| 24 | SAA+ | CVPR23 WS | CLIP+SAM | Zero-shot seg | - | Cascaded FM assembly |
| 25 | InCTRL | CVPR 2024 | Foundation | Few-shot | - | Residual learning with normal prompts |

**Legend**: MC=Multi-class, TF=Training-free, ZS=Zero-shot, FS=Few-shot, 1FA=One-for-all, WS=Workshop, HL=Highlight

---

## Key Trends and Observations

### 1. DINOv2 dominance for feature extraction
DINOv2 has become the de facto backbone for both reconstruction-based and nearest-neighbor methods. Its self-supervised patch-level features provide richer structural information than CLIP's visual features.

### 2. Training-free paradigm gaining traction
SubspaceAD, AnomalyDINO, UniVAD, SuperAD, LogSAD all demonstrate that frozen FM features + simple scoring can match or beat trained methods. This suggests FM features are already highly discriminative for anomaly detection.

### 3. CLIP vs DINOv2 specialization
- **CLIP**: Better for zero-shot (text-guided), semantic understanding, cross-domain transfer
- **DINOv2**: Better for few-shot/full-shot, fine-grained structural anomalies, patch-level features
- **Hybrid**: Emerging trend of combining both (LogSAD, CLIP-DINOv2 Fusion)

### 4. Multi-class/unified is the new standard
Single-class models are being replaced by unified multi-class approaches (Dinomaly, UniVAD, UniAD). Dinomaly2 shows a single model can handle all settings.

### 5. Logical anomaly detection remains unsolved
Most FM-based methods excel at structural anomalies but struggle with logical ones (e.g., wrong count, wrong arrangement). LogSAD and UniVAD attempt this with LLM/graph reasoning.

### 6. Performance ceiling approaching on MVTec AD
With Dinomaly2 at 99.9% I-AUROC, MVTec AD is near saturation. Real-IAD, MVTec AD 2, and domain-shift benchmarks are becoming the new evaluation frontier.

---

## Research Gaps (Opportunities)

1. **Domain shift robustness**: Most methods assume test distribution matches training. Real-world lighting, camera, and positioning changes are underexplored.
2. **Feature space understanding**: Why do FM features work so well for AD? Mechanistic understanding is lacking.
3. **Nuisance variation disentanglement**: How to separate normal variation (pose, lighting) from true anomalies in FM feature space.
4. **Calibration**: Anomaly scores from FM-based methods are poorly calibrated across categories.
5. **Efficiency-accuracy tradeoff**: Many SOTA methods require large ViT backbones; practical deployment needs efficiency.
6. **Logical anomaly detection with FMs**: Still a major gap -- structural approaches dominate.

---

## Related Notes
- analysis: `../analysis/fm_ad_robustness/`
- experiments: `../experiments/2026-03-27_ttns_go_nogo/`, `../experiments/2026-03-28_ifr_go_nogo/`
- ideas: `../ideas/2026-03-26_ASAD_proposal.md`
