# FM-Internal Nuisance Identification: 4 Ideas Deep Analysis (2026-03-28)

> **Context**: Paired NSP gives +29.4pp but requires paired data. TTNS (test-time unpaired via covariance shift) completely failed due to anomaly contamination. We need methods that leverage DINOv2's INTERNAL structure to identify nuisance directions WITHOUT external paired data or domain augmentation.
>
> **Key constraints**:
> - No paired multi-condition data
> - No domain-specific augmentation
> - Must avoid anomaly contamination (the TTNS killer)
> - Linear hard projection is optimal (10 nonlinear variants all failed)

---

## Executive Summary

| Idea | Theoretical Soundness | Key Risk | Expected Gain | Novelty | Priority |
|------|----------------------|----------|---------------|---------|----------|
| **1. IFR (Inter-layer Residual)** | Strong | Residuals may encode anomaly info too | +10-18pp over baseline | High | **1st** |
| **2. Cross-layer Covariance Contrast** | Moderate | Representation space mismatch | +5-12pp | Medium | 3rd |
| **3. Attention-weighted Scoring** | Strong | Attention doesn't track nuisance | +3-8pp | Low | 4th |
| **4. Cross-layer Disagreement** | Moderate-Strong | Agreement metric design | +8-15pp | Medium-High | **2nd** |
| **5. Residual Stream Decomposition** (NEW) | Strong | Complexity | +12-20pp | **Very High** | See below |

**Recommendation**: Prioritize Idea 1 (IFR) as primary, Idea 4 (Disagreement) as secondary. They can be combined. A novel Idea 5 (Residual Stream Decomposition) is proposed as the highest-novelty option.

---

## Idea 1: Inter-layer Feature Residual (IFR)

### Concept

DINOv2 uses residual connections: `f_L = f_{L-1} + attention_L(f_{L-1}) + ffn_L(...)`. The residual `Delta_L = f_L - f_{L-1}` represents what layer L *adds* to the representation.

Given that Layer 8 is naturally robust (RelShift 0.11) and Layer 11 is not (RelShift 0.22-0.27), the increments Delta_9, Delta_10, Delta_11 represent what the deep layers add beyond the robust bottleneck. The hypothesis is that these increments primarily encode nuisance-sensitive information.

**Algorithm**:
```
1. Extract train normal features at layers 8, 9, 10, 11
2. Compute residuals: Delta_L = f_L - f_{L-1} for L in {9, 10, 11}
3. Concatenate or pool: R = [Delta_9; Delta_10; Delta_11] or R = Delta_9 + Delta_10 + Delta_11
4. PCA on R across train normals -> top-K eigenvectors = nuisance directions
5. Project f_11 to remove these directions
6. Mahalanobis scoring on projected f_11
```

### 1. Theoretical Soundness: STRONG

This idea is well-grounded in two theoretical frameworks:

**A. Residual Stream Interpretation (Elhage et al., 2021)**
Mechanistic interpretability views the transformer as writing to a shared "residual stream." Each layer's contribution is additive and can be decomposed. The residual `Delta_L` isolates what that specific layer contributes. This is not an approximation -- it is exact due to the additive residual connection.

**B. Information Bottleneck at Layer 8**
If Layer 8 is the information bottleneck (minimal intrinsic dimension, maximum compression), then layers 9-11 must be *re-expanding* the representation. The V-shaped robustness profile (robust at L8, fragile at L11) suggests this re-expansion adds shift-sensitive features back. The residuals capture exactly this re-expansion.

**C. Causal reasoning**: If f_8 is robust and f_11 is not, and f_11 = f_8 + sum(Delta_L), then sum(Delta_L) must contain the nuisance component. This is a direct algebraic consequence, not an assumption.

**Key mathematical property**: Because DINOv2 uses residual connections with the same dimensionality throughout (d=768 per token), all layers live in the SAME vector space. This means `f_11 - f_8` is a well-defined vector, and PCA on these differences is mathematically valid. This is a critical advantage over Idea 2.

### 2. Key Assumption and What Would Break It

**Assumption**: The residuals Delta_{9-11} primarily encode nuisance-sensitive information, not anomaly-discriminative information.

**What would break it**: If layers 9-11 add information that is BOTH shift-sensitive AND anomaly-discriminative. For example, if deep layers refine texture encoding that is simultaneously useful for detecting scratches AND sensitive to lighting. The entanglement measured at L11 (PC correlation 0.53) suggests this partial overlap exists.

**Mitigation**: The anomaly signal at L8 alone is already strong (L8 gives 96.4% on clean MVTec AD). The question is whether removing deep-layer additions hurts the REMAINING anomaly signal. If most anomaly-discriminative information is already present at L8, the residuals are safe to remove.

**Quantitative test**: Compare L8-only scoring vs L11-after-IFR-projection scoring. If IFR makes L11 perform like L8 on clean data while improving on shifted data, the assumption holds.

### 3. Expected Effect Size

**Upper bound**: NSP at +29.4pp uses perfect paired shift vectors.
**Lower bound**: L8 alone (without projection) already gains +3.6pp over L11 by being naturally robust.

IFR is estimating nuisance directions from an indirect signal (layer residuals) rather than direct shift observations. The nuisance subspace estimate will be noisier than NSP's paired estimate, but:
- It uses NO external data (train normals only)
- The residual stream decomposition is exact (no statistical estimation error from test batch)
- No anomaly contamination (computed only from train normals)

**Estimate**: +10-18pp over baseline on AD2, achieving 70-78% (vs NSP's 83.8%). This would be 60-75% of NSP's gain. On clean AD1, expect minimal degradation (< -1pp) since L8 information is preserved.

**Reasoning**: The residuals capture a superset of the nuisance directions (they also contain some useful deep-layer refinements). So the projection will be slightly over-aggressive but in the right general direction.

### 4. Genuinely FM-Internal?

**Yes, strongly so.** This method exploits the specific architectural property of residual connections in ViTs. It would not work for:
- CNNs (different spatial resolutions across layers)
- Architectures without residual connections
- Models where intermediate layers are inaccessible

The method literally decomposes the FM's internal computation into per-layer contributions and uses the decomposition to identify nuisance. This is as FM-internal as it gets.

### 5. Novelty Assessment

**Novelty: HIGH**

The residual stream decomposition has been studied in NLP mechanistic interpretability (Elhage et al., 2021; logit prisms), but applying it to:
- Vision Transformers specifically
- Anomaly detection
- Nuisance identification via cross-layer subtraction

...has NOT been done. The closest work is:
- Dinomaly: uses multi-layer features but for reconstruction, not for nuisance decomposition
- DenseFormer (2024): weighted fusion of layer outputs, but for training, not analysis
- SINDER: analyzes per-layer artifacts but focuses on high-norm tokens, not nuisance subspaces

**Paper-worthy angle**: "The layers added beyond the information bottleneck encode nuisance, and removing them restores robustness" is a clean, testable claim with strong theoretical backing.

---

## Idea 2: Cross-layer Covariance Contrast

### Concept

Directions with high variance at L11 but low variance at L8 are candidates for nuisance (amplified by deep layers). Formalized as a generalized eigenvalue problem:

```
Sigma_L11 v = lambda * Sigma_L8 v
```

High lambda eigenvectors = directions where L11 has disproportionately more variance = nuisance candidates.

### 1. Theoretical Soundness: MODERATE

**The core mathematical question: Are L8 and L11 in the same space?**

For DINOv2 (and all standard ViTs with residual connections), YES. The residual connection `f_L = f_{L-1} + delta_L` means all layer outputs share the same d-dimensional space. Unlike CNNs where deeper layers have different spatial dimensions and channel counts, ViT layers all produce vectors in R^d.

However, while they share the same vector space, the EFFECTIVE subspaces used by each layer can differ substantially. L8 features may concentrate in a 50-dimensional subspace while L11 features concentrate in a different 80-dimensional subspace. The generalized eigendecomposition compares variance in shared directions, which is valid, but the interpretation requires care.

**Generalized eigendecomposition (GED) is well-established** in signal processing for exactly this purpose -- comparing covariance structures between two conditions. It finds directions that maximally discriminate the two covariance matrices. The mathematical framework (Cohen, 2022 tutorial) is sound.

**Connection to the problem**: If nuisance encoding increases variance in certain directions between L8 and L11, GED will find these. But this assumes nuisance is the PRIMARY source of variance increase, which may not hold.

### 2. Key Assumption and What Would Break It

**Assumption**: Directions with amplified variance from L8 to L11 are nuisance-encoding.

**What would break it**:
- Deep layers may also amplify anomaly-relevant directions (e.g., L11 may add texture detail that helps defect detection)
- Normal variation that is NOT nuisance may also be amplified (e.g., legitimate object appearance variation)
- The spectral gap between nuisance-amplified and anomaly-amplified directions may be too small to separate

**Critical issue**: Unlike IFR, this method does not distinguish WHAT was added by deep layers. It only sees the NET variance difference. If L11 has higher variance in a direction because it encodes richer object identity (not nuisance), removing that direction would hurt.

### 3. Expected Effect Size

**Weaker than IFR**. The fundamental issue is that variance amplification is a weaker signal than direct layer residuals. Not all variance increases are nuisance, and GED cannot distinguish the sources.

**Estimate**: +5-12pp over baseline on AD2 (65-72%). This is 20-40% of NSP's gain. Risk of clean performance degradation (-2 to -5pp on AD1) because some amplified directions carry useful anomaly information.

### 4. Genuinely FM-Internal?

**Partially.** The method uses multi-layer features (FM-internal) but the covariance contrast itself is a generic statistical technique. The FM-specific insight is "different layers have different robustness properties," but the analysis technique could be applied to any two representations.

### 5. Novelty Assessment

**Novelty: MEDIUM**

GED for source separation is a decades-old technique (EEG analysis, signal processing). Applying it to cross-layer ViT features for AD is novel in the specific application, but the method itself is well-known. XCiT (NeurIPS 2021) already uses cross-covariance in ViTs, though for different purposes.

The contribution would be "showing that GED on cross-layer covariances identifies nuisance directions in FM features" -- interesting but incremental.

---

## Idea 3: Attention-weighted Patch Scoring

### Concept

Use DINOv2's CLS-to-patch attention weights to modulate patch anomaly scores:
- High CLS-attention patches = content-relevant = weight UP in scoring
- Low/diffuse attention patches = background/nuisance = weight DOWN
- Per-image, no batch statistics, no anomaly contamination

### 1. Theoretical Soundness: STRONG (for a different problem)

**DINO's attention maps are known to segment objects** (Caron et al., ICCV 2021). The CLS token's attention to patch tokens acts as a soft segmentation mask, highlighting semantically important regions. This is well-validated.

**However, this addresses a different problem than nuisance direction identification.** Attention weighting addresses SPATIAL nuisance (which patches are background) rather than FEATURE-DIRECTION nuisance (which feature dimensions encode lighting/viewpoint). The distribution shift problem in FM-AD is primarily a feature-direction problem:
- Lighting changes affect ALL patches, not just background ones
- Viewpoint changes affect the feature vectors of content-relevant patches too
- The +29.4pp NSP gain comes from removing feature DIRECTIONS, not spatial locations

**Where attention weighting DOES help**: Reducing false positives from background patches that shift under lighting. AnomalyDINO already uses PCA-based background masking for similar reasons. The attention version is a per-image alternative.

### 2. Key Assumption and What Would Break It

**Assumption**: Nuisance effects are spatially concentrated in low-attention (background) regions.

**What would break it**: If lighting/viewpoint changes affect the feature vectors of content-relevant (high-attention) patches -- which they do. A globally overexposed image shifts features of foreground patches just as much as background ones. The attention map cannot distinguish "this patch's feature shifted due to lighting" from "this patch's feature shifted due to anomaly."

### 3. Expected Effect Size

**Modest**: +3-8pp over baseline on AD2 (63-68%). This addresses spatial noise but not the core feature-direction problem. On AD1, might help slightly (+0.5-1pp) by reducing background false positives.

**Key comparison**: AnomalyDINO already achieves 96.6% on MVTec AD using attention-based patch selection. But their problem is few-shot, not robustness. For robustness under shift, the gain will be smaller because the dominant failure mode is feature-level, not spatial.

### 4. Genuinely FM-Internal?

**Yes, but weakly.** Using the FM's own attention maps is FM-internal. However, using attention for spatial weighting is already done by AnomalyDINO and CLIP-DINOv2 fusion methods. The novelty is not in the mechanism but in the application to robustness.

### 5. Novelty Assessment

**Novelty: LOW**

Multiple existing works already use attention for patch weighting in AD:
- AnomalyDINO (WACV 2025): PCA-based background masking on DINOv2 features
- CLIP-DINOv2 (2024): Stabilized Attention-based Pooling
- AttentDifferNet (WACV 2024): Attention modules for anomaly scoring
- "Breaking the Bias" (2024): Recalibrating attention for AD

The specific idea of using CLS-attention as a robustness mechanism is incremental over these works.

---

## Idea 4: Self-referential Cross-layer Disagreement

### Concept

Compare L8 and L11 representations per-image to detect shift:
- Clean normal: L8 and L11 agree (both say "normal")
- Shifted normal: L11 deviates from L8 (nuisance added by deep layers)
- Anomaly: BOTH L8 and L11 flag it (true anomaly signal persists across layers)

Use L8-L11 disagreement as a shift indicator. If disagreement is high, rely more on L8 (robust); if low, use L11 (richer features).

### 1. Theoretical Soundness: MODERATE-STRONG

**The intuition is well-supported by the data**:
- L8: RelShift 0.11, nuisance-anomaly correlation 0.12 (naturally disentangled)
- L11: RelShift 0.22-0.27, correlation 0.53 (entangled)
- This means L11 adds nuisance-sensitive directions that L8 lacks

**Formalization**: Define a per-sample disagreement score:
```
d(x) = ||P_anomaly(f_8(x)) - P_anomaly(f_11(x))||
```
where P_anomaly projects onto the anomaly-discriminative subspace. But we don't know P_anomaly.

**Simpler proxy**: Use Mahalanobis distance at each layer:
```
s_8(x) = Mahal(f_8(x); mu_8, Sigma_8)
s_11(x) = Mahal(f_11(x); mu_11, Sigma_11)
disagreement(x) = |s_8(x) - s_11(x)|
```

**The logic**:
- For true anomalies: both s_8 and s_11 should be high -> low disagreement, high scores
- For shifted normals: s_11 inflated by nuisance, s_8 stable -> high disagreement, mixed scores
- For clean normals: both low -> low disagreement, low scores

**Adaptive scoring**:
```
final_score(x) = alpha(x) * s_8(x) + (1 - alpha(x)) * s_11(x)
alpha(x) = sigma(beta * disagreement(x))   # shift to L8 when disagreement is high
```

### 2. Key Assumption and What Would Break It

**Assumption**: Disagreement between L8 and L11 is primarily caused by nuisance (shift), not by anomaly type.

**What would break it**: If certain anomaly types cause large L8-L11 disagreement (e.g., anomalies that only appear at certain semantic levels). In that case, the method would incorrectly downweight L11 for those anomalies and rely on L8 which might miss them.

**Also**: The method is per-sample adaptive, which is powerful but harder to calibrate. The beta parameter and the functional form of alpha(x) need careful tuning.

**Critical risk**: This is fundamentally a SCORING modification, not a subspace projection. The 10 failed variants in Phase 3 included scoring modifications (dual-space ensemble: -9.2pp). The "linear hard projection is optimal" lesson may apply here too.

### 3. Expected Effect Size

**Moderate**: +8-15pp over baseline on AD2 (68-75%). The adaptive layer selection should help on shifted images (use L8) while preserving performance on clean images (use L11). But it does not REMOVE nuisance -- it AVOIDS it by layer selection.

**Key insight**: This is conceptually different from projection-based methods. It does not clean L11 features; it dynamically selects which layer to trust. The upper bound is L8-only performance on shifted data (which already captures most of the gain).

**On clean AD1**: Should maintain or slightly improve performance since alpha -> 0 (use L11) when no shift detected.

### 4. Genuinely FM-Internal?

**Yes, strongly.** This method fundamentally depends on:
- Multi-layer feature extraction (FM architecture)
- Layer-specific robustness properties (discovered empirically for DINOv2)
- The residual connection ensuring L8/L11 comparability

It would not work for single-layer feature extractors or models without the V-shaped robustness profile.

### 5. Novelty Assessment

**Novelty: MEDIUM-HIGH**

Per-sample adaptive layer selection based on cross-layer consistency is novel for AD. Related but distinct work:
- Multi-layer feature fusion is common (Dinomaly, PatchCore), but static
- Test-time layer selection based on confidence is studied in efficient inference, not AD
- The specific mechanism (robustness disagreement as shift detector) is new

**Risk**: A reviewer might say "this is just a mixture of two models with adaptive weights" -- technically true but the insight about WHY to mix (V-shaped robustness) and WHEN (disagreement = shift) is the contribution.

---

## Idea 5 (NEW): Residual Stream Decomposition for Nuisance Attribution

### Concept

This is a deeper version of Idea 1 that fully decomposes the DINOv2 computation:

```
f_L = f_0 + sum_{l=1}^{L} [attn_l + ffn_l]
```

Each layer contributes two terms: attention output and FFN output. We can decompose f_11 into 2*11 + 1 = 23 additive components. The question is: WHICH of these components encode nuisance?

**Algorithm**:
```
1. Extract all 23 components for train normals: {embed, attn_1, ffn_1, ..., attn_11, ffn_11}
2. For each component c_l, compute its contribution to the total variance:
   Var_contribution(c_l) = trace(Cov(c_l)) / trace(Cov(f_11))
3. For each component, compute its contribution to shift sensitivity:
   Shift_contribution(c_l) = how much c_l changes between conditions
   (estimated from train normals under different NATURAL within-train variation)
4. Components with high Shift_contribution / Var_contribution ratio = nuisance
5. Construct nuisance subspace from PCA on concatenated nuisance components
6. Project f_11 to remove
```

**Key insight**: This does NOT need paired data because it uses the model's own internal decomposition. The shift sensitivity can be estimated from within-train variation (different normal samples) rather than cross-condition variation.

### Why This Might Work Without Paired Data

The fundamental insight: **different layers/components of DINOv2 respond differently to input variation**. Even within a single environment, some components (e.g., attn layers 9-11) may encode more viewpoint/lighting-sensitive features than others (e.g., attn layers 4-8).

We can detect this from normal train data alone:
- Components whose variance is primarily explained by low-frequency spatial patterns (position, global illumination) = likely nuisance
- Components whose variance captures high-frequency local patterns (texture, edges) = likely content

The mechanistic interpretability literature shows that different layers/heads encode different types of information. Leveraging this structure for AD robustness is the novel contribution.

### Theoretical Soundness: STRONG

This directly extends the residual stream framework from NLP interpretability to vision. The decomposition is mathematically exact (no approximation). The attribution of components to nuisance vs content uses within-distribution statistics only (no test contamination).

### Key Risk

Complexity of implementation and the fact that within-train variation may not be a good proxy for cross-condition shift. If all conditions in training look the same (single lighting, single viewpoint), within-train variation does not reveal what would change under different conditions.

### Novelty: VERY HIGH

No existing work decomposes ViT features into per-layer-per-module (attention/FFN) contributions for the purpose of nuisance identification in AD. This bridges mechanistic interpretability and anomaly detection -- two fields that have not been connected.

---

## Comparative Analysis: Why These Methods Avoid the TTNS Failure Mode

The critical question: TTNS failed because test batch statistics are contaminated by anomalies. How do these 4+1 ideas avoid this?

| Method | Uses test statistics? | Anomaly contamination risk | Why it avoids TTNS failure |
|--------|----------------------|---------------------------|---------------------------|
| IFR | **No** (train only) | **None** | Nuisance estimated from layer residuals on train normals |
| Covariance Contrast | **No** (train only) | **None** | Compares train covariances across layers |
| Attention Scoring | **Per-image** | **None** | Attention is computed per-image by the FM itself |
| Disagreement | **Per-image** | **Low** | Per-sample score, no batch aggregation |
| Residual Decomposition | **No** (train only) | **None** | Uses model internals, not data statistics |

**This is the fundamental advantage**: All proposed methods derive nuisance information from the model's internal structure or per-image computation, NOT from test batch statistics. TTNS failed precisely because it relied on test batch statistics (Sigma_test) which are contaminated.

---

## Cross-Idea Synergies and Combinations

### Combination A: IFR + Disagreement (Recommended)

```
1. Use IFR to estimate nuisance directions from layer residuals
2. Project L11 features using IFR-derived nuisance subspace
3. Use L8 vs projected-L11 disagreement for adaptive scoring
```

This combines the subspace cleaning of IFR with the adaptive scoring of Disagreement. IFR handles the feature-direction problem; Disagreement handles residual shift that IFR misses.

### Combination B: IFR + Attention (Conservative)

```
1. Use attention maps to weight patches (spatial filtering)
2. Use IFR to clean features (direction filtering)
3. Score on attention-weighted, IFR-cleaned features
```

Two orthogonal types of nuisance removal (spatial + directional).

### Combination C: Full Residual Decomposition + Disagreement (Ambitious)

```
1. Decompose f_11 into 23 components
2. Identify nuisance-attributed components
3. Reconstruct f_11 minus nuisance components
4. Adaptive scoring with L8 as fallback
```

Highest potential but also highest implementation complexity.

---

## Practical Prioritization

### Priority 1: IFR (Inter-layer Feature Residual)

**Why first**:
- Simplest to implement (just layer subtraction + PCA)
- Strongest theoretical backing (exact decomposition)
- Highest expected effect size for the effort
- No hyperparameters beyond K (same as NSP)
- Direct comparison with NSP: same projection framework, different nuisance estimation
- Clean narrative: "What deep layers add beyond the bottleneck is nuisance"

**Quick experiment (1 day)**:
```python
# Extract L8, L9, L10, L11 features for train normals
# Compute Delta_L = f_L - f_{L-1} for L in {9, 10, 11}
# PCA on stacked Deltas -> nuisance directions
# Project L11, score with Mahalanobis
# Compare: baseline (59.9%) vs IFR vs NSP (83.8%)
```

**GO/NO-GO**:
- IFR > 65% on AD2 -> GO (nuisance directions are meaningful)
- IFR > 70% on AD2 -> STRONG GO (approaching NSP territory without paired data)
- IFR < 60% on AD2 -> NO-GO (residuals don't capture nuisance well)

### Priority 2: Cross-layer Disagreement

**Why second**:
- Complementary to IFR (scoring vs subspace)
- Per-image adaptive (no batch statistics)
- Can be layered on top of IFR

**Quick experiment (0.5 day after IFR)**:
```python
# Compute Mahalanobis scores at L8 and L11
# Measure disagreement per sample
# Adaptive alpha mixing
# Compare with IFR alone and IFR + Disagreement
```

### Priority 3: Covariance Contrast

**Why third**: Weaker signal, higher risk of removing useful directions. Try only if IFR disappoints.

### Priority 4: Attention Weighting

**Why last**: Addresses spatial problem, not the core feature-direction problem. Can be added as a minor improvement on top of other methods.

### Priority 5 (Research Moonshot): Residual Stream Decomposition

**Why separate track**: Highest novelty and potential paper impact, but highest implementation complexity. Start as a parallel analysis track while running IFR experiments.

---

## Paper Positioning If IFR Works

```
Title: "Anatomy of a Failure: How Foundation Model Layers Encode Nuisance
        in Visual Anomaly Detection"

Section 1: FM-AD fails under distribution shift (3 methods x 3 benchmarks, -25-34pp)
Section 2: V-shaped robustness profile across layers (empirical discovery)
           Layer 8 = information bottleneck, naturally disentangled
Section 3: What deep layers add is nuisance (IFR analysis)
           Residual stream decomposition: Delta_{9-11} encode shift-sensitive info
           PCA on residuals -> nuisance subspace (NO paired data needed)
Section 4: Method: IFR projection + optional cross-layer disagreement scoring
Section 5: Experiments
           - IFR vs NSP (oracle) vs baseline
           - IFR on MVTec AD 2 + RobustAD + clean MVTec AD
           - Ablation: which layers' residuals matter most
           - Analysis: correlation between Delta variance and shift sensitivity
Section 6: Discussion
           - "Why simple linear works" explained via residual decomposition
           - Limitations: assumes nuisance is in deep-layer additions
           - Connection to mechanistic interpretability
```

**Contribution**:
1. Discovery: V-shaped robustness profile in DINOv2 features
2. Analysis: Residual stream decomposition reveals nuisance encoding in deep layers
3. Method: IFR -- training-free, paired-data-free nuisance projection using only model internals
4. Evidence: Negative results (10 failed variants, TTNS failure) explained by the framework

---

## Related Notes

- [TTNS GO/NO-GO](../../experiments/2026-03-27_ttns_go_nogo/report.md) -- why test-time estimation failed
- [NSP -> SAPP methodology](2026-03-27_methodology_nsp_to_sapp.md) -- current best method details
- [4-team thought experiment](../2026-03-27_method_thought_experiment.md) -- previous idea evaluation
- [Mechanism analysis](2026-03-23_mechanism_analysis.md) -- 7 failure mechanisms
- [Theoretical foundations](2026-03-23_theoretical_foundations.md) -- information theory backbone
- [SEAS proposal](2026-03-26_shift_equivariant_scoring_proposal.md) -- transport-based alternative

## References (Search-derived)

- Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits" -- residual stream framework
- SINDER (ECCV 2024, arXiv 2407.16826) -- DINOv2 high-norm token artifacts, per-layer analysis
- AnomalyDINO (WACV 2025) -- DINOv2 patch-based AD with background masking
- XCiT (NeurIPS 2021) -- cross-covariance in vision transformers
- Cohen (2022) -- tutorial on generalized eigendecomposition for source separation
- DINOv2 Register Tokens (ICLR 2024) -- artifact tokens and layer behavior
- Dinomaly (CVPR 2025) -- multi-layer DINOv2 reconstruction for AD
