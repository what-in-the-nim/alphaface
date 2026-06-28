# AlphaFace: High Fidelity and Real-time Face Swapper Robust to Facial Pose

**arXiv**: [2601.16429v1](https://arxiv.org/abs/2601.16429v1)

**Authors**: Jongmin Yu¹,², Hyeontaek Oh¹, Zhongtian Sun³, Angelica I Aviles-Rivero⁴, Moongu Jeon⁵, Jinhong Yang¹,⁶

**Affiliations**:
1. ProjectG.AI
2. University of Cambridge
3. University of Kent
4. Tsinghua University
5. Gwangju Institute of Science and Technology
6. Inje University

**Contact**: jy522@projectg.ai

---

## Abstract

Existing face-swapping methods often deliver competitive results in constrained settings but exhibit substantial quality degradation when handling extreme facial poses. This paper introduces AlphaFace, which leverages vision-language models (VLMs) and CLIP embeddings to apply semantic contrastive losses. The approach achieves real-time performance (~24.1ms per frame, ~41.5 FPS) while maintaining robustness to extreme poses without explicit geometric priors. Results on FF++, MPIE, and LPFF datasets demonstrate superior performance compared to state-of-the-art methods.

---

## 1. Introduction

Face swapping replaces facial identity in images/videos while preserving non-identity attributes (lighting, hairstyle, pose, expression). While valuable for entertainment and creativity, the technique raises ethical concerns regarding identity misuse and non-consensual content.

**Central challenge**: Accurately transferring source identity while preserving target image attributes — particularly problematic under extreme facial poses.

Existing approaches struggle with:
- Significant angular variations causing facial geometry distortion
- Self-occlusions and boundary misalignment
- Computational costs prohibiting real-time applications (diffusion methods: 46+ seconds per image)

AlphaFace addresses these limitations through semantic supervision from vision-language models rather than explicit geometric priors.

---

## 2. Related Work

### GAN/Autoencoder-Based Methods

Prior approaches (FaceShifter, FSGAN, SimSwap/SimSwap++) follow a two-step pipeline:
1. Extract identity features from source using independent encoder
2. Merge features with target attributes to generate swapped face

Research focuses on:
- Improving source identity representation
- Preserving target-specific attributes (illumination, texture, accessories, poses)

Advanced architectures include semantic-guided fusion layers and identity injection blocks with various learning strategies.

### Pose-Robustness Approaches

- **HiFiFace**: Incorporates 3D Morphable Models for local texture warping
- **FaceDancer**: Proposes interpretability-based regularization
- **Diffusion-based methods**: Achieve photorealism but remain impractical for real-time use and still struggle with extreme poses (±60°+)

### Distinction of AlphaFace

Unlike geometry-dependent methods, AlphaFace leverages "rich semantic information obtained from a vision-language model during training" without requiring explicit facial geometry information.

---

## 3. AlphaFace

### 3.1 Architecture

The framework comprises three principal modules:

#### 1. Source Identity Encoder

Uses ArcFace to extract discriminative latent features $c_s$. Ensures generated output preserves desired identity. Follows prior face-swapping work for robust, generalizable identity representation.

#### 2. Fusion Encoder — Cross-Adaptive Identity Injection (CAII) Block

The CAII block integrates source and target features while preserving target attributes.

**AdaIN on target features**:

$$\text{AdaIN}(z_t, \varphi(c_s)) = \sigma(\varphi(c_s)) \cdot \frac{z_t - \mu(z_t)}{\sigma(z_t)} + \mu(\varphi(c_s))$$

where:
- $\mu$ = mean computation function
- $\sigma$ = standard deviation computation function
- $\varphi$ = neural network mapping source identity to target-compatible latent space

Output: $\hat{z}_t = \alpha(\text{Conv}(\text{AdaIN}(z_t, \varphi(c_s))))$

**AdaIN on source identity features**:

$$\hat{z}_s = \text{AdaIN}(\varphi(c_s), z_t) + \varphi(c_s)$$

This normalizes source features while attenuating irrelevant information.

**Identity injection**:

$$\bar{z}_t = (\hat{z}_t \otimes \hat{z}_s) \oplus \hat{z}_s$$

where $\otimes$ = element-wise multiplication, $\oplus$ = element-wise summation.

#### 3. Face Generator

Progressively upsamples fused latent representation through deconvolutional layers to produce high-resolution outputs preserving source identity and target attributes.

---

### 3.2 Objective Functions

AlphaFace training combines five loss components:

#### Loss 1: Identity Swap Loss $\mathcal{L}^{t \to s}_{ID}$

$$\mathcal{L}^{t \to s}_{ID} = 1 - \frac{c_s \cdot c_{t \to s}}{\|c_s\|_2 \|c_{t \to s}\|_2}$$

Encourages swapped image identity similarity to source face using cosine angular similarity based on ArcFace features.

#### Loss 2: Attribute Preservation Loss $\mathcal{L}^{t \to s}_{AP}$

Combines three components:

**Masked reconstruction loss**:
$$\mathcal{L}^{t \to s}_{Rec}(x_{t \to s}, x_t) = \|(1 - m_t) \otimes (x_{t \to s} - x_t)\|_1$$

**Cyclic reconstruction loss**:
$$\mathcal{L}^{t \to s \to t}_{Cycle}(x_{t \to s \to t}, x_t) = \|x_{t \to s \to t} - x_t\|_1$$

**Perceptual loss**: VGG16 deep features providing semantics-aware gradients.

**Combined**:
$$\mathcal{L}^{t \to s}_{AP} = \mathcal{L}^{t \to s}_{Rec} + \mathcal{L}^{t \to s \to t}_{Cycle} + \mathcal{L}^{t \to s}_{Percept}$$

The masked reconstruction learns non-facial region information; cycle loss addresses boundary/background precision essential for pose robustness.

#### Loss 3: Adversarial Learning Loss $\mathcal{L}^{t \to s}_{Adv}$

PatchGAN discriminator enhances visual quality by restoring high-frequency details (edge acuity, fine texture) on small image patches.

#### Loss 4: CLIP Image-to-Text Loss $\mathcal{L}^{t \to s}_{CLIP\text{-}text}$

$$\mathcal{L}^{t \to s}_{CLIP\text{-}text} = \tau \left(1 - \frac{\langle \varphi_{img}(x_{t \to s}),\, \varphi_{text}(t_t) \rangle}{\|\varphi_{img}(x_{t \to s})\| \cdot \|\varphi_{text}(t_t)\|}\right)$$

where:
- $\varphi_{img}$, $\varphi_{text}$ = CLIP image and text encoders
- $t_t$ = text description of target face (generated via OpenGVLab/InternVL3-14B)
- $\tau$ = validity indicator (1 if swapped image is less consistent with target description than original)

**Caption prompt used**:
> "Describe pose, background, facial accessories, and all obstacles covering the face area in the given face image. Only 70 words are allowed."

#### Loss 5: CLIP Identity Swapping Loss $\mathcal{L}^{t \to s}_{CLIP\text{-}ID}$

$$\mathcal{L}^{t \to s}_{CLIP\text{-}ID} = 1 - \frac{\langle \varphi_{img}(x_{t \to s}),\, \varphi_{img}(x_s) \rangle}{\|\varphi_{img}(x_{t \to s})\| \cdot \|\varphi_{img}(x_s)\|}$$

Reinforces source identity representation using visual features.

#### Total Objective

$$\mathcal{L}^{t \to s}_{Total} = \lambda_{ID}\mathcal{L}^{t \to s}_{ID} + \lambda_{AP}\mathcal{L}^{t \to s}_{AP} + \lambda_{Adv}\mathcal{L}^{t \to s}_{Adv} + \lambda_{CLIP}(\mathcal{L}^{t \to s}_{CLIP\text{-}text} + \mathcal{L}^{t \to s}_{CLIP\text{-}ID})$$

---

## 4. Experiments

### 4.1 Setup

**Training data**: VGGFace2-HQ, CelebA-HQ

**Evaluation datasets**:
- FaceForensics++ (FF++)
- Multi-Pose Illumination Expressions (MPIE)
- Large-Pose Flickr Face (LPFF)

**Preprocessing**: Source images 112×112, target images 256×256

**Metrics**:
- Identity proximity (cosine similarity, CSIM)
- Identity-retrieval accuracy
- Pose error
- Expression error
- Fréchet Inception Distance (FID)
- Inference speed (ms)

**Hyperparameters**:
- $\lambda_{ID}=10.0$, $\lambda_{AP}=0.5$, $\lambda_{Adv}=1.0$, $\lambda_{CLIP}=1.0$
- Batch size: 8
- Optimizer: Adam, initial LR: 0.01 (decayed ×0.9 every 5 epochs)
- Total epochs: 50
- Hardware: 2× A6000 (training), 1× RTX 4090 (testing)

---

### 4.2 Ablation: CLIP-Based Losses

**FF++ dataset**:

| Setting | ID Retrieval ↑ | Pose Error ↓ | Expr Error ↓ | FID ↓ |
|---------|---------------|-------------|-------------|-------|
| w/o CLIPs | 96.82 | 2.75 | 3.82 | 4.95 |
| CLIP w/o text | 97.67 | 2.07 | 2.58 | 2.90 |
| CLIP w/o ID | 98.52 | 1.58 | 2.19 | 3.12 |
| **w CLIPs (ours)** | **98.77** | **1.24** | **2.03** | **2.71** |

**MPIE dataset**:

| Setting | CSIM ↑ | Pose Error ↓ | Expr Error ↓ | FID ↓ |
|---------|--------|-------------|-------------|-------|
| w/o CLIPs | 0.427 | 4.19 | 5.03 | 11.04 |
| CLIP w/o text | 0.465 | 3.82 | 3.43 | 8.12 |
| CLIP w/o ID | 0.467 | 3.12 | 3.17 | 9.61 |
| **w CLIPs (ours)** | **0.471** | **2.97** | **3.03** | **7.78** |

**Key finding**: CLIP-text loss provides a stronger single contribution than CLIP-ID loss. Combined losses show complementary but sub-additive improvements, indicating partially overlapping supervisory signals. Textual supervision supplies identity-agnostic semantic constraints that better suppress pose/expression errors whilst preserving source identity.

---

### 4.3 Ablation: Identity Injection

**FF++ dataset**:

| Approach | ID Retrieval ↑ | Pose Error ↓ | Expr Error ↓ | FID ↓ |
|----------|---------------|-------------|-------------|-------|
| Unidirectional | 98.80 | 1.27 | 2.68 | 5.27 |
| **CAII (ours)** | 98.77 | **1.24** | **2.03** | **2.71** |

**MPIE dataset**:

| Approach | CSIM ↑ | Pose Error ↓ | Expr Error ↓ | FID ↓ |
|----------|--------|-------------|-------------|-------|
| Unidirectional | 0.452 | 3.41 | 4.18 | 10.9 |
| **CAII (ours)** | **0.471** | **2.97** | **3.03** | **7.78** |

CAII maintains identity preservation while reducing pose/expression errors and improving visual quality, particularly under extreme pose variations.

---

### 4.4 Comparison with State-of-the-Art

**FF++ benchmark**:

| Method | ID Retrieval ↑ | Pose Error ↓ | Expr Error ↓ | FID ↓ | Speed (ms) ↓ |
|--------|---------------|-------------|-------------|-------|-------------|
| FSGAN | 61.07 | 3.31 | 3.02 | 15.36 | 21.5 |
| SimSwap | 93.01 | 1.53 | 2.84 | 7.48 | 27.1 |
| BlendFace | 97.02 | 3.07 | 2.14 | 3.84 | 24.7 |
| HiFiFace | 98.01 | 2.84 | 2.51 | 10.25 | 22.3 |
| FaceDancer | 98.84 | 2.04 | 7.97 | 16.30 | 78.3 |
| DiffSwap | 98.54 | 2.45 | 5.35 | **2.16** | 46245.2 |
| **AlphaFace** | **98.77** | **1.24** | **2.03** | 2.71 | **24.1** |

**MPIE dataset**:

| Method | CSIM ↑ | Pose Error ↓ | Expr Error ↓ | FID ↓ |
|--------|--------|-------------|-------------|-------|
| FSGAN | 0.105 | 5.31 | 4.02 | 43.64 |
| SimSwap | 0.180 | 3.92 | 3.81 | 16.89 |
| BlendFace | 0.392 | 3.71 | 3.18 | 11.27 |
| HiFiFace | 0.092 | 5.01 | 4.65 | 12.68 |
| FaceDancer | 0.401 | 4.72 | 3.31 | 10.54 |
| DiffSwap | 0.278 | 4.58 | 4.12 | 12.63 |
| **AlphaFace** | **0.471** | **2.97** | **3.03** | **7.78** |

**Analysis**:
- AlphaFace achieves the best pose/expression error metrics across both datasets
- FaceDancer achieves the highest FF++ ID retrieval (98.84) but has large expression error (7.97) and is slow (78.3ms = ~12.8 FPS)
- DiffSwap offers better FID on FF++ (2.16) but at 46.2 seconds per image — impractical for real-time use
- AlphaFace runs at ~41.5 FPS, making it suitable for real-time applications
- AlphaFace leads all metrics on the extreme-pose MPIE dataset

---

## 5. Conclusion

AlphaFace achieves robust face-swapping under extreme poses by combining:

1. Rich semantic supervision from open-source vision-language models
2. CLIP-informed contrastive losses for text-image alignment
3. Cross-Adaptive Identity Injection (CAII) for target-adaptive source representation
4. Competitive architecture maintaining real-time performance (~40 FPS)

The method surpasses state-of-the-art on pose/expression metrics across FF++, MPIE, and LPFF benchmarks while maintaining practical inference speed.

**Limitations**: Empirical selection of OpenGVLab/InternVL3-14B without comprehensive ablation on prompt variations or caption quality. Caption noise sensitivity and pose/expression-specific prompts are not yet investigated.

**Future work**: In-depth analysis of VLM caption quality, ablations on caption noise and prompt variations (pose-only, expression-only, combined).

---

## Key Contributions

1. Novel face-swapping framework leveraging VLM semantic supervision and CLIP encoders with contrastive losses
2. Cross-Adaptive Identity Injection (CAII) module improving identity representation isolation from unnecessary attributes
3. Open-source baseline with state-of-the-art performance and practical real-time capability
4. Empirical validation across multiple datasets demonstrating VLM supervision effectiveness

---

**Project repository**: https://github.com/andrewyu90/Alphaface_Official.git
