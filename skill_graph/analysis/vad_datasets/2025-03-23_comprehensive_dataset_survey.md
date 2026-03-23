# Comprehensive Survey: Visual Anomaly Detection Benchmark Datasets

**Date**: 2025-03-23
**Scope**: All major VAD benchmark datasets (classic through 2025)
**Purpose**: Research reference for dataset selection and positioning

---

## Summary Table

| Dataset | Year | Venue | Images | Categories | Resolution | Annotation | Modality |
|---------|------|-------|--------|------------|------------|------------|----------|
| DAGM | 2007 | DAGM Symposium | ~16,100 | 10 | 512x512 | Ellipse (weak) | RGB (synthetic) |
| AITEX | 2019 | AFID paper | 245 | 7 fabrics | 4096x256 | Pixel mask | RGB |
| KolektorSDD | 2019 | JIM | 399 | 1 (commutators) | ~500x1240 | Pixel mask | RGB |
| MVTec AD | 2019 | CVPR | 5,354 | 15 | 700x700~1024x1024 | Pixel mask + Image | RGB |
| BTAD | 2021 | ISIE | 2,830 | 3 | 600x600~1600x1600 | Pixel mask | RGB |
| KolektorSDD2 | 2021 | COMIND | 3,336 | 1 (commutators) | ~230x630 | Pixel mask (instance) | RGB |
| MPDD | 2022 | arXiv | 1,346 | 6 | Various | Pixel mask | RGB |
| MVTec 3D-AD | 2021 | VISAPP (Best Paper) | 4,147 | 10 | High-res | Pixel mask | RGB + 3D Point Cloud |
| MVTec LOCO AD | 2022 | IJCV | 3,644 | 5 | 850x1700 | Pixel mask | RGB |
| VisA | 2022 | ECCV | 10,821 | 12 | ~1000x1000+ | Pixel mask + Image | RGB |
| Eyecandies | 2022 | ACCV | ~15,000 | 10 | High-res | Pixel mask | RGB + Depth + Normal (synthetic) |
| DTD-Synthetic | 2023 | Various | ~2,400 | 12 textures | Various | Pixel mask | RGB (synthetic) |
| Real3D-AD | 2023 | NeurIPS | 1,254 | 12 | High-precision 3D | Point-level | 3D Point Cloud |
| PAD (MAD dataset) | 2023 | NeurIPS | 11,000+ | 20 (LEGO toys) | High-res | Pixel mask | RGB |
| AeBAD | 2023 | CIM | 5,570 | 1 (blades) | Various | Pixel mask | RGB + Video |
| GoodsAD | 2024 | IEEE RA-L | 6,124 | 6 | 3000x3000 | Pixel mask + Image | RGB |
| Real-IAD | 2024 | CVPR | 150,000 | 30 | High-res | Pixel mask + Image | RGB (multi-view, 5 angles) |
| CableInspect-AD | 2024 | NeurIPS | 4,798 | 3 (cables) | High-res | Pixel mask + Severity | RGB |
| Texture-AD | 2024 | arXiv | 43,120 | 39 (cloth/wafer/metal) | Various | Pixel mask | RGB |
| RAD (Robustness) | 2024 | ECCV | Various | 4 | Various | Pixel mask | RGB |
| RAD (Multi-view) | 2024 | arXiv | Various | 13 | Various | Pixel mask | RGB |
| MMAD | 2024 | ICLR 2025 | 8,366 + 39,672 Q&A | 38 (from 4 datasets) | Various | MCQ annotations | RGB + Text |
| RobustAD | 2025 | CVPR-W | Various | Multiple | Various | Pixel mask | RGB |
| MVTec AD 2 | 2025 | arXiv / VAND | 8,000+ | 8 | 2.6-5 MP | Pixel mask (public+private split) | RGB |
| Real-IAD D3 | 2025 | CVPR | Large-scale | 20 | High-res | Pixel mask | RGB + 3D + Pseudo-3D |
| Real-IAD Variety | 2025 | Pattern Recognition | 198,950 | 160 | High-res | Pixel mask + Image | RGB |
| MANTA | 2025 | CVPR | 137,338 | 38 (5 domains) | High-res | Pixel mask + Text descriptions | RGB + Text (multi-view) |
| ISP-AD | 2025 | J. Intell. Manufacturing | 559,049 | Screen printing | Various | Weak labels | RGB |
| HSS-IAD | 2025 | ICME | 8,580 | Metallic parts | Various | Pixel mask | RGB |

---

## Detailed Dataset Profiles

---

### 1. DAGM 2007
- **Full Name**: DAGM 2007 Competition Dataset for Optical Inspection
- **Year / Venue**: 2007 / DAGM Symposium
- **Images**: ~16,100 (14,000 defect-free + 2,100 defective)
- **Categories**: 10 synthetically generated texture classes
- **Resolution**: 512x512
- **Data Characteristics**:
  - Synthetically generated textures with artificially placed defects
  - Each class has 1,000 normal images and 150 defective images
  - One defect per defective image
- **Unique Features**: One of the earliest defect detection benchmarks. Weakly supervised setting -- labels are ellipses roughly outlining defective areas, not pixel-precise masks
- **Annotation**: Weak labels (elliptical bounding regions)
- **Evaluation**: Classification accuracy, segmentation
- **SOTA**: Near-saturated performance (>99% on most classes)
- **Availability**: Public, available on Kaggle

---

### 2. AITEX (AFID)
- **Full Name**: AITEX Fabric Image Database
- **Year / Venue**: 2019 / AFID paper
- **Images**: 245 (140 defect-free + 105 defective)
- **Categories**: 7 different fabric types
- **Resolution**: 4096x256 (elongated strips)
- **Data Characteristics**:
  - Textile/fabric defect detection
  - Various weave patterns and fabric types
  - Defects include holes, broken threads, contamination
- **Unique Features**: Very elongated aspect ratio (16:1). Small dataset, focused specifically on textile quality inspection
- **Annotation**: Pixel-level binary masks
- **Evaluation**: Accuracy, precision, recall
- **SOTA**: ~98% accuracy (SCUNet)
- **Availability**: Public, from aitex.es/afid/

---

### 3. KolektorSDD (KSDD)
- **Full Name**: Kolektor Surface-Defect Dataset
- **Year / Venue**: 2019 / Journal of Intelligent Manufacturing
- **Images**: 399 (347 defect-free + 52 defective)
- **Categories**: 1 (electrical commutators)
- **Resolution**: ~500x1240
- **Data Characteristics**:
  - Real-world industrial images of commutator surfaces
  - 50 items, 8 non-overlapping images per item
  - Very small dataset, highly imbalanced (13% defective)
- **Unique Features**: Captured in controlled industrial environment. Very challenging due to small size and class imbalance. Uses 3-fold cross-validation
- **Annotation**: Pixel-level segmentation masks
- **Evaluation**: AP (Average Precision), per-image classification
- **Availability**: Public, CC BY-NC-SA 4.0, from vicos.si

---

### 4. KolektorSDD2 (KSDD2)
- **Full Name**: Kolektor Surface-Defect Dataset 2
- **Year / Venue**: 2021 / Computers in Industry
- **Images**: 3,336 (2,332 train + 1,004 test)
- **Categories**: 1 (electrical commutators, improved version)
- **Resolution**: ~230x630
- **Data Characteristics**:
  - Significantly expanded from original KolektorSDD
  - 394 labeled defect instances
  - Real production line images
- **Unique Features**: Supports mixed supervision (weak to full). Instance segmentation annotations available
- **Annotation**: Pixel-level instance segmentation masks
- **Evaluation**: AP, classification accuracy
- **Availability**: Public, CC BY-NC-SA 4.0, from vicos.si

---

### 5. MVTec AD
- **Full Name**: MVTec Anomaly Detection Dataset
- **Year / Venue**: 2019 / CVPR (extended IJCV 2021)
- **Images**: 5,354 (3,629 train + 1,725 test)
- **Categories**: 15 (5 textures + 10 objects)
  - Textures: carpet, grid, leather, tile, wood
  - Objects: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper
- **Resolution**: 700x700 to 1024x1024
- **Data Characteristics**:
  - 70+ different defect types across categories
  - Defects: scratches, dents, contaminations, structural changes
  - Mix of texture and object categories
  - Single-instance per image, well-aligned
- **Unique Features**: THE gold standard benchmark for VAD. First comprehensive real-world industrial AD dataset. Well-controlled acquisition. Near-saturated performance by current methods
- **Annotation**: Pixel-level binary masks + Image-level labels
- **Evaluation Protocol**:
  - Image-level: AUROC
  - Pixel-level: AUROC, AU-PRO (per-region overlap)
  - Standard: unsupervised (train on normal only)
- **Current SOTA**:
  - Image AUROC: ~99.6% (Dinomaly, multi-class) / ~99.8% (SimpleNet, single-class)
  - Pixel AUROC: ~98.9%
  - **Performance is near-saturated** -- methods compete within <1% range
- **Availability**: Public, CC BY-NC-SA 4.0, from mvtec.com

---

### 6. BTAD
- **Full Name**: BeanTech Anomaly Detection Dataset
- **Year / Venue**: 2021 / ISIE (VT-ADL paper)
- **Images**: 2,830 (2,540 train + 290 test)
- **Categories**: 3 industrial products
- **Resolution**: Product-dependent (600x600 to 1600x1600)
- **Data Characteristics**:
  - Real-world industrial products from beanTech srl
  - Varying image sizes per product class
  - Product 1: 400 train, Product 2: 1000 train, Product 3: 399 train
- **Unique Features**: Introduced with VT-ADL (Vision Transformer for AD). Smaller but realistic industrial dataset. Product categories not publicly named (proprietary)
- **Annotation**: Pixel-level masks for each anomalous image
- **Evaluation**: AUROC (image + pixel)
- **Availability**: Public, on Kaggle and GitHub

---

### 7. MPDD
- **Full Name**: Metal Parts Defect Detection Dataset
- **Year / Venue**: 2022 / arXiv
- **Images**: 1,346 (888 train normal + 176 test normal + 282 test abnormal)
- **Categories**: 6 metal part types
- **Resolution**: Various (around 1024x1024)
- **Data Characteristics**:
  - Painted metal parts from real production lines
  - Various spatial orientations, positions, distances
  - Different light intensities and non-uniform backgrounds
  - Defects from manufacturing process
- **Unique Features**: Challenging due to uncontrolled conditions (varying pose, lighting, background). More realistic than well-aligned datasets. Also has MPDD2 extension (~700 images)
- **Annotation**: Pixel-level masks
- **Evaluation**: AUROC (image + pixel)
- **Availability**: Public, on GitHub (stepanje/MPDD)

---

### 8. MVTec 3D-AD
- **Full Name**: MVTec 3D Anomaly Detection Dataset
- **Year / Venue**: 2021 / VISAPP (Best Industrial Paper Award)
- **Images/Scans**: 4,147 high-resolution 3D point cloud scans + aligned RGB images
- **Categories**: 10 objects
  - Natural variation: bagel, carrot, cookie, peach, potato
  - Deformable: foam, rope, tire
  - Rigid: cable gland, dowel
- **Resolution**: High-resolution 3D + RGB
- **Data Characteristics**:
  - Industrial 3D sensor acquisition
  - Defects: scratches, dents, holes, contaminations, deformations
  - Both RGB and 3D modalities aligned
  - Considerable natural variation in some categories
- **Unique Features**: First large-scale 3D AD benchmark. Multimodal (RGB + 3D). Categories span rigid to deformable objects with natural variation
- **Annotation**: Pixel-precise ground truth for anomalous regions (both 2D and 3D)
- **Evaluation**: AUROC, AU-PRO (both image and pixel level, both modalities)
- **Availability**: Public, CC BY-NC-SA 4.0, from mvtec.com

---

### 9. MVTec LOCO AD
- **Full Name**: MVTec Logical Constraints Anomaly Detection Dataset
- **Year / Venue**: 2022 / IJCV
- **Paper**: "Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization"
- **Images**: 3,644 (1,772 train + 304 validation + 1,568 test)
- **Categories**: 5
  - Breakfast box, juice bottle, pushpins, screw bag, splicing connectors
- **Resolution**: ~850x1700
- **Data Characteristics**:
  - **Structural anomalies**: scratches, dents, contaminations (traditional)
  - **Logical anomalies**: wrong count, misplacement, missing parts, wrong arrangement
  - Anomalies violate logical/functional constraints rather than just texture
- **Unique Features**: First dataset to explicitly separate structural vs logical anomalies. Logical anomalies require understanding of object relationships and constraints, not just local texture. Critical gap for VLM-based approaches
- **Annotation**: Pixel-precise masks for all anomalous regions
- **Evaluation**: AU-PRO (separate for structural vs logical), AUROC
- **Current SOTA**: Logical anomalies remain significantly harder than structural ones
- **Availability**: Public, CC BY-NC-SA 4.0, from mvtec.com

---

### 10. VisA
- **Full Name**: Visual Anomaly Dataset
- **Year / Venue**: 2022 / ECCV
- **Paper**: "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation"
- **Images**: 10,821 (9,621 normal + 1,200 anomalous)
- **Categories**: 12, grouped into 3 domains:
  - PCBs (4): PCB1, PCB2, PCB3, PCB4
  - Multi-instance (4): Capsules, Candles, Macaroni1, Macaroni2
  - Single-instance (4): Cashew, Chewing gum, Fryum, Pipe fryum
- **Resolution**: ~1000x1000+ (varies by category)
- **Data Characteristics**:
  - Complex objects (PCBs with many components)
  - Multi-instance scenes (multiple objects per image)
  - Surface defects: scratches, dents, color spots, cracks
  - Structural defects: misplacement, missing parts
- **Unique Features**: Largest industrial AD dataset at the time. Multi-instance challenge (Capsules, Macaroni). PCB categories with complex structures. From Amazon Science
- **Annotation**: Image-level + Pixel-level masks
- **Evaluation**: AUROC (image + pixel), AP
- **Current SOTA**: Image AUROC ~98.7% (Dinomaly, multi-class)
- **Availability**: Public, on AWS Open Data Registry

---

### 11. Eyecandies
- **Full Name**: The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization
- **Year / Venue**: 2022 / ACCV
- **Images**: ~15,000+ (10 classes, train/val/test splits)
- **Categories**: 10 candy types (procedurally generated)
- **Resolution**: High-resolution
- **Data Characteristics**:
  - Fully synthetic (photo-realistic rendering)
  - Procedurally generated candies on conveyor belt
  - 948 anomalous test objects, 41 defect types
  - Multiple lighting conditions per sample
- **Unique Features**: Synthetic multimodal dataset (RGB + Depth + Normal maps). Arbitrary scalability via procedural generation. Controlled lighting variations. Unique modality combination rarely available in other datasets
- **Annotation**: Pixel-precise ground truth for test set
- **Evaluation**: AUROC, AU-PRO
- **Availability**: Public, from eyecan-ai.github.io/eyecandies

---

### 12. DTD-Synthetic
- **Full Name**: Describable Textures Dataset - Synthetic (for anomaly detection)
- **Year / Venue**: 2023 / Various (used in DRAEM, DeSTSeg, etc.)
- **Images**: ~2,400 (12 textures, ~100 train + ~100 test each)
- **Categories**: 12 texture types (subset of original 47-category DTD)
- **Resolution**: Various
- **Data Characteristics**:
  - Synthetic anomalies generated by blending texture patches
  - Based on the Describable Textures Dataset (DTD, 5,640 images, 47 categories)
  - Used as anomaly source for methods like DRAEM
- **Unique Features**: Bridge between texture classification and anomaly detection. DTD is widely used as a source of synthetic anomaly textures (Perlin noise + DTD patches)
- **Annotation**: Pixel-level masks (synthetic)
- **Note**: DTD itself is primarily used as an auxiliary dataset for anomaly synthesis, not as a standalone AD benchmark

---

### 13. Real3D-AD
- **Full Name**: Real3D-AD: A Dataset of Point Cloud Anomaly Detection
- **Year / Venue**: 2023 / NeurIPS (Datasets & Benchmarks Track)
- **Samples**: 1,254 high-resolution 3D items
- **Categories**: 12 (Airplane, Car, Candybar, Chicken, Diamond, Duck, Fish, Gemstone, Seahorse, Shell, Starfish, Toffees)
- **Resolution**: 0.0010mm-0.0015mm point cloud precision; 40K to millions of points per item
- **Data Characteristics**:
  - Pure 3D point cloud (no RGB)
  - 360-degree coverage
  - Real-world objects with real defects
  - High-precision industrial-grade scanning
- **Unique Features**: Highest point cloud resolution among 3D AD datasets. 360-degree complete coverage. Pure point cloud benchmark (unlike MVTec 3D-AD which is RGB+3D). Baseline method Reg3D-AD proposed
- **Annotation**: Point-level anomaly labels
- **Evaluation**: AUROC (point-level and object-level)
- **Availability**: Public, on GitHub (M-3LAB/Real3D-AD)

---

### 14. PAD / MAD (Multi-pose Anomaly Detection)
- **Full Name**: PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection
- **Year / Venue**: 2023 / NeurIPS (Datasets & Benchmarks Track)
- **Images**: 11,000+ high-resolution RGB images
- **Categories**: 20 (complex-shaped LEGO toys)
- **Resolution**: High-resolution
- **Data Characteristics**:
  - MAD-Sim: simulated environment with synthetic anomalies
  - MAD-Real: real-world environment with real defects
  - 4K+ views with various poses per object
  - 3 types of anomalies: scratch, dent, color anomaly
- **Unique Features**: First benchmark for pose-agnostic AD. Objects appear in arbitrary poses (not aligned). Tests whether methods can separate pose variation from real defects. 2025 extension: PIAD adds illumination-agnostic challenge
- **Annotation**: Pixel-precise ground truth
- **Evaluation**: AUROC, 8 paradigms and 10 methods benchmarked
- **Availability**: Public, on GitHub (EricLee0224/PAD)

---

### 15. AeBAD
- **Full Name**: Aero-engine Blade Anomaly Detection Dataset
- **Year / Venue**: 2023 / Computers in Industry
- **Paper**: "Industrial Anomaly Detection with Domain Shift"
- **Images**: 5,570 (1,228 normal + 4,342 abnormal)
- **Categories**: 1 (aero-engine blades), 2 sub-datasets:
  - AeBAD-S: single blade images at different scales
  - AeBAD-V: video of blades on rotating blisks
- **Resolution**: Various
- **Data Characteristics**:
  - Domain shift between train and test distributions
  - Changes in illumination and viewpoint
  - Unaligned samples at different scales
  - 4 defect types: breakdowns, ablations, grooves, fractures
- **Unique Features**: Explicitly addresses domain shift problem in AD. Samples are NOT aligned and at different scales. Proposed MMR (Masked Multi-scale Reconstruction) baseline
- **Annotation**: Pixel-level masks
- **Evaluation**: AUROC, with domain shift evaluation
- **Availability**: Public, on GitHub (zhangzilongc/MMR)

---

### 16. GoodsAD (PKU-GoodsAD)
- **Full Name**: PKU-GoodsAD: A Supermarket Goods Dataset for Unsupervised Anomaly Detection and Segmentation
- **Year / Venue**: 2024 / IEEE Robotics and Automation Letters
- **Images**: 6,124 (4,464 normal + 1,660 anomalous)
- **Categories**: 6 (484 different-appearance goods)
  - Boxed cigarettes, bottled drinks, canned drinks, bottled foods, boxed foods, packaged foods
- **Resolution**: 3000x3000
- **Data Characteristics**:
  - Supermarket/retail products (not industrial parts)
  - 8 anomaly types: deformation, surface damage, opened, wrong label, etc.
  - Real commercial products with consumer-facing defects
- **Unique Features**: First dataset focused on retail/supermarket product anomalies. High intra-class variation (484 different product appearances). Extends AD beyond industrial manufacturing to retail quality control
- **Annotation**: Image-level + Pixel-level masks
- **Evaluation**: AUROC (image + pixel), PRO
- **Availability**: Public, on GitHub (jianzhang96/GoodsAD)

---

### 17. Real-IAD
- **Full Name**: Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection
- **Year / Venue**: 2024 / CVPR
- **Images**: ~150,000
- **Categories**: 30 objects
- **Resolution**: High-definition
- **Data Characteristics**:
  - Multi-view: 5 shooting angles per sample
  - Materials: metal, plastic, wood, ceramics, mixed
  - Defects: missing parts, dirt, deformation, pits, damage, holes, cracks, scratches
  - Challenging defects closer to real production scenarios
- **Unique Features**: Order-of-magnitude larger than prior datasets. Multi-view acquisition (5 angles). Proposed sample-level evaluation metrics. Introduced FUIAD (Fully Unsupervised IAD) setting where yield rate >60%
- **Annotation**: Image-level + Pixel-level masks
- **Evaluation**:
  - Standard: AUROC, AU-PRO
  - Novel: sample-level metrics, FUIAD evaluation protocol
- **Current SOTA**: Image AUROC ~89.3% (Dinomaly, multi-class) -- still far from saturated
- **Availability**: Public, from realiad4ad.github.io

---

### 18. CableInspect-AD
- **Full Name**: CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset
- **Year / Venue**: 2024 / NeurIPS (Datasets & Benchmarks Track)
- **Images**: 4,798 (6,023 annotated anomalies)
- **Categories**: 3 types of power line cables
- **Resolution**: High-resolution
- **Data Characteristics**:
  - Power line cables from Hydro-Quebec (Canadian utility)
  - Expert-annotated by domain specialists
  - Different cable colors, textures, braiding patterns
  - Varying severity levels for each anomaly
  - Real-world robotic inspection scenario
- **Unique Features**: Expert-annotated (domain specialists, not crowdsourced). Severity-graded anomalies. Proposed Enhanced-PatchCore baseline. Evaluates few-shot, many-shot, and zero-shot (VLM) settings
- **Annotation**: Pixel-level masks + Severity levels
- **Evaluation**: AUROC with few-shot and zero-shot protocols
- **Availability**: Public, from mila-iqia.github.io/cableinspect-ad

---

### 19. Texture-AD
- **Full Name**: Texture-AD: An Anomaly Detection Dataset and Benchmark for Real Algorithm Development
- **Year / Venue**: 2024 / arXiv
- **Images**: 43,120 total
  - Cloth: 6,283 (15 types)
  - Semiconductor wafer: 14,861 (14 types)
  - Metal plate: 21,976 (10 types)
- **Categories**: 39 (across 3 material domains)
- **Resolution**: Various
- **Data Characteristics**:
  - Real manufacturing process defects (10+ types)
  - Scratches, wrinkles, color variations, point defects
  - Training and test data may come from DIFFERENT product specifications
- **Unique Features**: First dataset explicitly addressing the gap between development data and production data. Training/test from different product specs (unlike MVTec where train/test are same product). Highly challenging for SOTA methods
- **Annotation**: Pixel-level masks
- **Evaluation**: AUROC (image + pixel)
- **Availability**: Public, on Hugging Face

---

### 20. RAD (Robustness)
- **Full Name**: RAD: A Comprehensive Dataset for Benchmarking the Robustness of Image Anomaly Detection
- **Year / Venue**: 2024 / ECCV
- **Data Characteristics**:
  - Tests robustness under imaging noise
  - Free views, uneven illumination, blurry acquisition
  - Foreign objects on working platforms as anomalies
  - 11 SOTA methods benchmarked
- **Unique Features**: Systematic robustness evaluation. Separate evaluation of viewpoint, illumination, and blur effects
- **Evaluation**: AUROC with robustness-specific analysis
- **Availability**: Public, on GitHub (hustCYQ/RAD-dataset)

---

### 21. RAD (Pose-Agnostic Multi-View)
- **Full Name**: RAD: A Dataset and Benchmark for Real-Life Anomaly Detection with Robotic Observations
- **Year / Venue**: 2024 / arXiv
- **Categories**: 13 everyday objects
- **Data Characteristics**:
  - 68 diverse robotic viewpoints per category
  - Uncontrolled illumination
  - 4 defect types: scratched, missing, stained, squeezed
  - Reflective materials, textureless surfaces, geometric symmetries
- **Unique Features**: Robotic observation perspective. Explicitly tests pose-agnostic detection with realistic distribution shift

---

### 22. MMAD
- **Full Name**: MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection
- **Year / Venue**: 2024 / ICLR 2025
- **Images**: 8,366 from 38 product classes (aggregated from 4 public datasets)
- **Questions**: 39,672 multiple-choice questions across 7 subtasks
- **Data Characteristics**:
  - Not a new image dataset but a benchmark protocol
  - Tests MLLMs (GPT-4o, LLaVA, etc.) on AD tasks
  - 7 subtasks: defect existence, type, location, severity, etc.
- **Unique Features**: First MLLM benchmark for industrial AD. Tests language-guided anomaly understanding. GPT-4o achieves only 74.9% average accuracy -- far from industrial requirements
- **Annotation**: Multiple-choice Q&A pairs
- **Evaluation**: Accuracy on 7 subtasks
- **Availability**: Public, on GitHub (jam-cc/MMAD) and Hugging Face

---

### 23. RobustAD
- **Full Name**: Robust AD: A Real World Benchmark Dataset For Robustness in Industrial Anomaly Detection
- **Year / Venue**: 2025 / CVPR Workshop (VAND 3.0)
- **Data Characteristics**:
  - Multiple industries: pharmaceutical, automotive, semiconductor
  - 9 anomaly types across 8 test domains
  - Controlled distribution shifts across multiple dimensions
- **Unique Features**: Novel "Average Relative Drop" metric measuring performance degradation under domain shifts. From Amazon Science. Tests real-world robustness beyond controlled lab settings
- **Evaluation**: Average Relative Drop (ARD), AUROC
- **Availability**: Public, on HuggingFace (AmazonScience/RobustAD)

---

### 24. MVTec AD 2
- **Full Name**: MVTec Anomaly Detection 2 Dataset
- **Year / Venue**: 2025 / arXiv (VAND 3.0 Challenge dataset)
- **Images**: 8,000+
- **Categories**: 8 (Can, Fabric, Fruit Jelly, Rice, Sheet Metal, Vial, Wallplugs, Walnuts)
- **Resolution**: 2.6-5 megapixels (high-resolution)
- **Data Characteristics**:
  - Transparent and overlapping objects (Vial)
  - Mirror-like reflections (Sheet Metal)
  - High normal variance (Walnuts, Fabric)
  - Dark-field and back-light illumination
  - Extremely small defects
  - Defects at image borders
  - Multiple lighting conditions per scene
- **Unique Features**: Successor to MVTec AD addressing saturation problem. Split into public and private test sets. Private test evaluated only via benchmark server. Multiple lighting conditions test robustness to distribution shift. AU-PRO_0.05 (strict threshold) used for evaluation
- **Annotation**: Pixel-precise masks (public test set only; private test evaluated by server)
- **Evaluation Protocol**:
  - AU-PRO_0.05 (integrated to FPR=0.05 only)
  - Public evaluation server at benchmark.mvtec.com
  - Default input: 256x256
- **Current SOTA**: <31% AU-PRO_0.05 at default 256x256; <60% AU-PRO at higher resolution
- **Availability**: Public download, CC BY-NC-SA 4.0, from mvtec.com. Evaluation server for private test

---

### 25. Real-IAD D3
- **Full Name**: Real-IAD D3: A Real-World 2D/Pseudo-3D/3D Dataset for Industrial Anomaly Detection
- **Year / Venue**: 2025 / CVPR
- **Categories**: 20
- **Data Characteristics**:
  - Multimodal: RGB + 3D point cloud + Pseudo-3D (photometric stereo)
  - Industrial components with smaller dimensions and finer defects
  - Micrometer-level 3D point cloud precision
  - Diverse anomalies across modalities
- **Unique Features**: First to include pseudo-3D modality (photometric stereo). Combines strengths of RGB, point cloud, and pseudo-3D depth. Tests uni-modal vs multi-modal approaches
- **Annotation**: Pixel-level masks
- **Evaluation**: AUROC, AU-PRO
- **Availability**: Public, from realiad4ad.github.io/Real-IAD_D3 and HuggingFace

---

### 26. Real-IAD Variety
- **Full Name**: Real-IAD Variety: Pushing Industrial Anomaly Detection Dataset to a Modern Era
- **Year / Venue**: 2025 / Pattern Recognition
- **Images**: 198,950
- **Categories**: 160
- **Data Characteristics**:
  - 28 industries, 24 material types, 22 color variations, 27 defect types
  - Unprecedented diversity and scale
  - Extension of Real-IAD from 30 to 160 categories
- **Unique Features**: Largest IAD dataset to date (by category count). Key finding: multi-class MUAD methods degrade 10-20% when scaling from 30 to 160 categories, but zero-shot/few-shot methods are robust to category scaling. Essential for training foundation models for AD
- **Annotation**: Image-level + Pixel-level masks
- **Evaluation**: AUROC, with scalability analysis
- **Availability**: Public

---

### 27. MANTA
- **Full Name**: MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects
- **Year / Venue**: 2025 / CVPR
- **Images**: 137,338 (8,600 anomalous with pixel-level annotations)
- **Categories**: 38 across 5 domains (Agriculture, Medicine, Electronics, Mechanics, Groceries)
- **Resolution**: High-resolution, 5 viewpoints per sample
- **Data Characteristics**:
  - Multi-view (5 distinct viewpoints per object)
  - Visual-text multimodal
  - Tiny object anomalies (heterogeneous, pose-agnostic, size-sensitive)
  - Spans beyond industrial to agriculture and medicine
  - Text component: 875 declarative knowledge words + 2K MCQ questions
- **Unique Features**: First large-scale multi-view + visual-text AD dataset. Tiny object focus (unique challenge). Spans 5 diverse domains. Text annotations describe anomalies with what/why/how. MCQ questions at varying difficulty levels
- **Annotation**: Pixel-level masks + Text descriptions (Declarative Knowledge + Constructivist Learning)
- **Evaluation**: AUROC, with visual-text task baselines
- **Availability**: Public

---

### 28. ISP-AD
- **Full Name**: Industrial Screen Printing Anomaly Detection Dataset
- **Year / Venue**: 2025 / Journal of Intelligent Manufacturing
- **Images**: 559,049 (312,674 normal + 245,664 synthetic defects + 711 real defects)
- **Categories**: Screen printing patterns
- **Resolution**: Various
- **Data Characteristics**:
  - Largest publicly available industrial AD dataset by image count
  - Small and weakly contrasted surface defects
  - Structured patterns with high permitted design variability
  - Mix of synthetic and real defects from factory floor
- **Unique Features**: Massive scale (559K images). Mixed supervision: synthetic + real defects. Tests cold-start strategy (synthetic defects first, then few real defects). Addresses gap between lab conditions and real production
- **Annotation**: Weak labels (not pixel-precise)
- **Evaluation**: Detection accuracy, mixed supervision protocols
- **Availability**: Public, on Kaggle (orvile/isp-ad-dataset)

---

### 29. HSS-IAD
- **Full Name**: HSS-IAD: A Heterogeneous Same-Sort Industrial Anomaly Detection Dataset
- **Year / Venue**: 2025 / ICME
- **Images**: 8,580
- **Categories**: Metallic-like industrial parts (heterogeneous same-sort)
- **Resolution**: Various
- **Data Characteristics**:
  - Parts with structural and appearance variations
  - Subtle defects resembling base materials
  - Very low anomalous pixel ratio (3.0%)
  - Foreground images provided for synthetic anomaly generation
- **Unique Features**: Addresses multi-class setting with heterogeneous parts from the same factory. Defects are visually similar to normal material (high similarity). Bridges gap between academic datasets and real factory conditions. Low defect-to-background contrast
- **Annotation**: Pixel-level masks
- **Evaluation**: AUROC, under multi-class and class-separated settings
- **Availability**: Public, on GitHub (Qiqigeww/HSS-IAD-Dataset)

---

### 30. COCO-AD
- **Full Name**: COCO-AD: General-Purpose Anomaly Detection Dataset
- **Year / Venue**: 2024 / arXiv (InvAD paper)
- **Data Characteristics**:
  - Derived from COCO panoptic segmentation dataset
  - Large-scale, general-purpose (not industrial-specific)
  - Real-life scenes with high complexity
- **Unique Features**: First general-purpose AD dataset (not limited to industrial). Tests whether industrial AD methods generalize to open-world scenarios. Proposed practical metrics: mF1, mAcc, mIoU variants
- **Annotation**: Panoptic segmentation masks repurposed for AD
- **Evaluation**: mF1^.2_.8, mAcc^.2_.8, mIoU^.2_.8, mIoU-max
- **Availability**: Public (derived from COCO)

---

### 31. Uni-Medical
- **Full Name**: Uni-Medical Anomaly Detection Dataset
- **Year / Venue**: 2024 / Used in ADer benchmark
- **Data Characteristics**:
  - Medical images converted from CT scans
  - Distribution differs significantly from industrial datasets
  - Multiple medical imaging categories
- **Unique Features**: Bridges industrial AD methods to medical imaging domain. Tests cross-domain transferability of AD algorithms
- **Annotation**: Image-level + Pixel-level
- **Evaluation**: AUROC (multi-class setting in ADer)
- **Note**: Also related to BMAD (Benchmarks for Medical Anomaly Detection, CVPR-W 2024) which covers 6 datasets from 5 medical domains

---

## Key Trends and Analysis

### Scale Evolution
```
2007 DAGM:         ~16K images, 10 categories
2019 MVTec AD:      5K images, 15 categories
2022 VisA:         11K images, 12 categories
2024 Real-IAD:    150K images, 30 categories
2025 Real-IAD Var: 199K images, 160 categories
2025 ISP-AD:      559K images, 1 domain (largest by count)
2025 MANTA:       137K images, 38 categories, 5 domains
```

### Modality Evolution
- **2007-2019**: RGB only
- **2021**: RGB + 3D point cloud (MVTec 3D-AD)
- **2022**: RGB + Depth + Normal maps (Eyecandies, synthetic)
- **2024-2025**: RGB + 3D + Pseudo-3D (Real-IAD D3), RGB + Text (MANTA, MMAD)

### Challenge Dimensions
1. **Saturation**: MVTec AD saturated (~99.6% AUROC) -> MVTec AD 2 (<60% AU-PRO)
2. **Domain shift**: AeBAD, RobustAD, Texture-AD test train/test distribution mismatch
3. **Pose variation**: PAD/MAD, RAD test pose-agnostic detection
4. **Logical anomalies**: MVTec LOCO AD tests counting, arrangement, missing parts
5. **Multi-class scaling**: Real-IAD Variety shows 10-20% degradation at 160 categories
6. **Tiny objects**: MANTA focuses on size-sensitive anomaly detection
7. **Multi-view**: Real-IAD (5 views), MANTA (5 views), RAD (68 views)
8. **VLM/MLLM evaluation**: MMAD, MANTA test language-guided anomaly understanding

### Performance Landscape (as of early 2025)
| Dataset | Best Image AUROC | Best Pixel AUROC | Status |
|---------|-----------------|-----------------|--------|
| MVTec AD | ~99.6-99.8% | ~98.9% | Near-saturated |
| VisA | ~98.7% | ~98%+ | Near-saturated |
| MVTec LOCO AD | ~85-90% | Lower for logical | Active research |
| Real-IAD | ~89.3% | -- | Challenging |
| MVTec AD 2 | -- | <60% AU-PRO | Very challenging |
| Real-IAD Variety | Degrades 10-20% vs Real-IAD | -- | Very challenging |

### Evaluation Metric Evolution
- **Classic**: Image AUROC, Pixel AUROC
- **Refined**: AU-PRO (per-region overlap, penalizes large connected FPs)
- **Strict**: AU-PRO_0.05 (only integrates up to 5% FPR)
- **Practical**: mF1, mAcc, mIoU (COCO-AD)
- **Robustness**: Average Relative Drop (RobustAD)
- **Sample-level**: Real-IAD sample-level metrics
- **Proposed**: AUPIMO (redefining localization benchmarks, low tolerance)

---

## Research Gaps and Opportunities

1. **Logical anomaly detection** remains far from solved (MVTec LOCO AD)
2. **Scalability** to 100+ categories with a single model is unsolved (Real-IAD Variety findings)
3. **Domain shift robustness** is a critical real-world gap (AeBAD, Texture-AD, RobustAD)
4. **Multimodal fusion** (RGB + 3D + text) is still early (Real-IAD D3, MANTA)
5. **MVTec AD 2** poses a fresh challenge with <60% SOTA performance
6. **VLM-based AD** shows promise but GPT-4o only reaches 74.9% on MMAD
7. **Few-shot/zero-shot** methods are more robust to category scaling than supervised approaches

---

## Related Notes
- `../vad_trend/` -- VAD methodology trends
- `../../papers/` -- Individual paper analyses

## Sources
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- MVTec AD 2: https://www.mvtec.com/company/research/datasets/mvtec-ad-2, arXiv:2503.21622
- MVTec LOCO AD: https://www.mvtec.com/company/research/datasets/mvtec-loco
- MVTec 3D-AD: https://www.mvtec.com/company/research/datasets/mvtec-3d-ad
- VisA: https://registry.opendata.aws/visa/, ECCV 2022
- Real-IAD: https://realiad4ad.github.io/Real-IAD/, CVPR 2024
- Real-IAD D3: https://realiad4ad.github.io/Real-IAD_D3/, CVPR 2025
- Real-IAD Variety: arXiv:2511.00540, Pattern Recognition 2025
- BTAD: VT-ADL paper, ISIE 2021
- MPDD: https://github.com/stepanje/MPDD
- DAGM: Kaggle (DAGM 2007 competition)
- KolektorSDD/KSDD: https://www.vicos.si/resources/kolektorsdd/
- KolektorSDD2/KSDD2: https://www.vicos.si/resources/kolektorsdd2/
- AITEX: https://www.aitex.es/afid/
- MANTA: arXiv:2412.04867, CVPR 2025
- RobustAD: HuggingFace (AmazonScience/RobustAD), CVPR-W 2025
- ISP-AD: arXiv:2503.04997, Kaggle
- HSS-IAD: arXiv:2504.12689, ICME 2025
- MMAD: arXiv:2410.09453, ICLR 2025
- GoodsAD: arXiv:2307.04956, IEEE RA-L 2024
- Real3D-AD: NeurIPS 2023
- PAD/MAD: arXiv:2310.07716, NeurIPS 2023
- AeBAD: arXiv:2304.02216
- CableInspect-AD: NeurIPS 2024
- Texture-AD: arXiv:2409.06367
- Eyecandies: ACCV 2022
- COCO-AD: arXiv:2404.10760
- Awesome Industrial AD: https://github.com/M-3LAB/awesome-industrial-anomaly-detection
