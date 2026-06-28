# AlphaFace

AlphaFace is a face identity swapping model designed to transfer the identity from a source face onto a target face while preserving the target pose, expression, background, occlusions, lighting, and other non-identity attributes.

The method combines an ArcFace-style identity encoder, an AlphaFace swapper/generator, a discriminator, CLIP image/text supervision, face masks, and VLM-generated captions. During training, the model learns from randomly paired faces: one image provides the target attributes and another image provides the source identity.

![AlphaFace Architecture](https://arxiv.org/html/2601.16429v1/x2.png)

## Repository Layout

```text
configs/
  train.yaml                         Default Lightning training config
  eval.yaml                          Default batch inference/evaluation config

src/alphaface/
  train_clip.py                      Lightning training entry point
  eval.py                            Batch source x target inference entry point
  infer.py                           Single source/target image inference helper
  lit_module.py                      Training loop, losses, optimizers, prediction step
  data_module.py                     Lightning datamodule
  dataset/get_dataloader.py          Training dataset readers
  preprocess/
    prepare_dataset.py               End-to-end dataset preparation CLI
    align.py                         InsightFace face detection and FFHQ alignment
    mask.py                          BiSeNet ONNX face mask generation
    caption.py                       OpenAI-compatible VLM caption client
    pack_png.py                      Packed PNG dataset format
```

The code expects an AlphaFace model implementation at `alphaface.models.swapper_alphaface`. Make sure that module is present in your checkout before training, evaluation, or ONNX export.

## Installation

Use Python 3.10 or newer. A CUDA-capable GPU is strongly recommended for training and for preparing large datasets.

### 1. Create an Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the Package

For normal training and inference:

```bash
pip install -e .
```

For preprocessing raw images into an AlphaFace dataset:

```bash
pip install -e ".[preprocess]"
```

The older requirements file is also available:

```bash
pip install -r requirements.txt
```

Prefer `pip install -e .` when working on this repository because it installs the `alphaface-*` console scripts declared in `pyproject.toml`.

### 3. Install PyTorch for Your Hardware

The project depends on PyTorch, but the best wheel depends on your CUDA, ROCm, MPS, or CPU environment. If the default install does not match your GPU, install PyTorch from the official selector first, then install this project:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[preprocess]"
```

Adjust the PyTorch index URL for your CUDA version.

### 4. External Model Files

Create the expected model directories:

```bash
mkdir -p models weights
```

Common files used by the configs and scripts:

| File | Used by | Purpose |
| --- | --- | --- |
| `models/alphaface_demo.pt` | `configs/eval.yaml` | Demo/evaluation AlphaFace swapper checkpoint. |
| `models/vit_b_fr_pgair.pt` | `configs/train.yaml` | Initial/resume checkpoint for the swapper and identity encoder. |
| `weights/resnet34.onnx` or `weights/resnet18.onnx` | dataset preparation | BiSeNet face parsing model for mask generation. |
| InsightFace `buffalo_l` weights | dataset preparation | Face detection and landmark alignment. Downloaded automatically by InsightFace on first use. |
| CLIP ViT-B/32 weights | training and packing | CLIP image/text embeddings. Downloaded by `openai-clip` on first use. |

The original demo README linked these checkpoints:

- AlphaFace demo model: [Google Drive](https://drive.google.com/file/d/18ZOQB3WmIFnMwi1GqBroFFEOuSNKWpZQ/view?usp=sharing)
- ArcFace/identity checkpoint: [Google Drive](https://drive.google.com/file/d/1qc4s6eRQPluma72WFibUnw74GPMAYRtY/view?usp=drive_link)

Place downloaded checkpoints at the paths referenced by your config, or update the config fields.

## Dataset Preparation

Training uses packed PNG samples under `packed/*.png`. Each RGBA PNG stores the aligned face image, mask, captioner text, CLIP image embedding, CLIP text embedding, and ArcFace identity embedding together. The dataloader requires this packed layout and fails fast if metadata is missing.

### Raw Input Images

Start with a directory of face images:

```text
raw_faces/
  person_a/
    image_001.jpg
    image_002.jpg
  person_b/
    frame_001.png
```

The preparation script searches recursively for common image extensions. By default, it keeps the largest detected face in each image. Use `--all-faces` if your raw images can contain multiple useful faces.

### Caption Server

The captioner calls an OpenAI-compatible vision API. The default is a local Ollama-style endpoint:

```text
http://localhost:11434/v1
```

The default model name is:

```text
llava:7b
```

Any compatible vision-language model can be used if it accepts image messages through `/chat/completions`.

### Prepare a Packed Dataset

```bash
alphaface-prepare-dataset \
  --input ./raw_faces \
  --output ./custom \
  --device cuda \
  --mask-model ./weights/resnet34.onnx \
  --caption-url http://localhost:11434/v1 \
  --caption-model llava:7b \
  --id-encoder-checkpoint ./models/vit_b_fr_pgair.pt \
  --embed-batch-size 64
```

Equivalent module invocation:

```bash
python -m alphaface.preprocess.prepare_dataset \
  --input ./raw_faces \
  --output ./custom \
  --device cuda \
  --mask-model ./weights/resnet34.onnx \
  --caption-url http://localhost:11434/v1 \
  --caption-model llava:7b \
  --id-encoder-checkpoint ./models/vit_b_fr_pgair.pt
```

After preparation, `./custom` should contain:

```text
custom/
  img/
    000000_person_a_image_001.png
  mask/
    000000_person_a_image_001.png
  txt/
    000000_person_a_image_001.txt
  packed/
    000000_person_a_image_001.png
```

Use `--delete-originals` if you want to keep only `packed/` after packing:

```bash
alphaface-prepare-dataset \
  --input ./raw_faces \
  --output ./custom \
  --device cuda \
  --mask-model ./weights/resnet34.onnx \
  --caption-url http://localhost:11434/v1 \
  --caption-model llava:7b \
  --id-encoder-checkpoint ./models/vit_b_fr_pgair.pt \
  --delete-originals
```

### Convert an Existing Dataset

If you already have:

```text
custom/
  img/*.png
  mask/*.png
  txt/*.txt
```

pack it with:

```bash
alphaface-pack-dataset \
  --dataset ./custom \
  --device cuda \
  --id-encoder-checkpoint ./models/vit_b_fr_pgair.pt \
  --embed-batch-size 64
```

If `packed/<stem>.png` already exists, the converter reuses any valid metadata
already present in that PNG. For example, a sample with
`alphaface_clip_img` but no `alphaface_clip_txt` or `alphaface_id_emb` keeps the
stored CLIP image embedding and computes only the missing fields.

Force specific fields to be recalculated with `--rebuild`:

```bash
alphaface-pack-dataset \
  --dataset ./custom \
  --device cuda \
  --id-encoder-checkpoint ./models/vit_b_fr_pgair.pt \
  --rebuild clip_image
```

Valid rebuild fields are `caption`, `clip_image`, `clip_text`, and `arcface`.
Use `--rebuild-all` to regenerate every packed metadata field.

## Dataset Format

Packed samples live under:

```text
custom/
  packed/
    sample_0001.png
    sample_0002.png
```

Each file is an RGBA PNG:

| Storage location | Content |
| --- | --- |
| RGB channels | Aligned face image. |
| Alpha channel | Face mask, with `0` for face and `255` for background. |
| PNG iTXt `alphaface_caption` | VLM caption text. |
| PNG iTXt `alphaface_clip_img` | CLIP ViT-B/32 image embedding, 512 float16 values encoded as base64. |
| PNG iTXt `alphaface_clip_txt` | CLIP ViT-B/32 text embedding, 512 float16 values encoded as base64. |
| PNG iTXt `alphaface_id_emb` | ArcFace identity embedding, 512 float16 values encoded as base64. |

During training, the dataloader randomly chooses two different packed samples. One acts as the target attribute image and the other acts as the source identity image.

## Training

The default training config is a Lightning CLI config at `configs/train.yaml`.

Before training, update the dataset and checkpoint paths:

```yaml
model:
  config:
    db_path: ./custom
    model_path: ./models/vit_b_fr_pgair.pt
    output: ./save/save_clip

data:
  db_path: ./custom
```

Then run:

```bash
alphaface-train
```

or:

```bash
python -m alphaface.train_clip
```

The shell helper does the same with `CUDA_VISIBLE_DEVICES=0`:

```bash
bash train_clip.sh
```

You can also pass a config explicitly:

```bash
alphaface-train fit --config configs/train.yaml
```

### Training Outputs

With the default config, training writes:

```text
logs/swap_clip/                  TensorBoard logs
save/save_clip/last.ckpt         Latest Lightning checkpoint
save/save_clip/model_step=*.ckpt Step checkpoints every 25,000 train steps
```

The checkpoint callback in `configs/train.yaml` is configured with:

```yaml
every_n_train_steps: 25000
save_last: true
save_top_k: -1
filename: model_{step}
```

Expected trained artifacts:

| Artifact | Meaning |
| --- | --- |
| `last.ckpt` | Full Lightning checkpoint for resuming training. |
| `model_step=*.ckpt` | Periodic Lightning checkpoints. |
| Swapper weights inside the checkpoint | The inference/export model weights used for face swapping. |
| Discriminator weights inside the checkpoint | Training-only adversarial model state. Not needed for inference. |
| Optimizer/scheduler state | Used when resuming training. Not needed for inference. |

For deployment, you usually need the inference path used by `AlphaFace.forward`: the trained swapper/generator and any identity-encoding components that forward path calls.

## Models Used During Training

AlphaFace training uses several models with different responsibilities.

### AlphaFace Swapper / Generator

The swapper is the deployable face-swapping network. It receives:

- A target attribute image at 256 x 256.
- A source identity image normalized as an ArcFace-style 112 x 112 tensor.

It returns a 256 x 256 swapped image. This is the model you keep for inference and export.

### ArcFace Identity Encoder

The identity encoder extracts a 512-dimensional identity embedding from the source face and from generated faces. It is used for identity preservation losses. Source identity embeddings are stored in packed PNG metadata. Generated images still pass through the identity encoder during training so the identity loss has gradients with respect to the generator output.

The identity encoder may be part of the deployed graph depending on the model implementation. The safest export path is the same path used by PyTorch inference: call the full AlphaFace model with `(target, source)`. Export the lower-level swapper submodule only after verifying that its forward signature matches the tensors you intend to serve.

### CLIP ViT-B/32 Image Encoder

CLIP image embeddings supervise semantic similarity. The CLIP image-to-image loss encourages generated images to match source identity-related visual semantics. Packed datasets store CLIP image embeddings for real images, while generated images still pass through CLIP during training.

CLIP is frozen and is not exported for normal swapper inference.

### CLIP ViT-B/32 Text Encoder

CLIP text embeddings are computed from the VLM captions. The CLIP text loss encourages the swapped output to preserve target attributes described in text, such as pose, background, accessories, and occlusions. Packed datasets can store these text embeddings in advance.

CLIP text is frozen and is not used in normal inference.

### Vision-Language Caption Model

The VLM is used during dataset preparation, not inside the training step itself. It produces the per-face captions stored in `txt/*.txt` or in packed PNG metadata. The default client talks to an OpenAI-compatible server and requests captions with a prompt focused on pose, background, facial accessories, and obstacles covering the face.

### Face Alignment Model

InsightFace detects faces and landmarks, then the preprocessing script aligns faces to an FFHQ-style 256 x 256 crop. This is dataset-preparation infrastructure, not a trainable AlphaFace component.

### Face Mask Model

The BiSeNet ONNX face parser generates binary masks. Masks are used by the masked reconstruction loss so the model can preserve target/background regions correctly.

### Discriminator

The discriminator is used only after `adv_sess` training steps. It provides adversarial supervision for sharper and more realistic generated faces. It is training-only and is not needed for inference or ONNX export.

### VGG/Perceptual Model

The model uses a perceptual loss module when available in the AlphaFace implementation. This frozen network provides feature-space reconstruction supervision.

## Inference

### Batch Inference

The default evaluation config reads source images from `./dataset/source`, target images from `./dataset/target`, and writes every source x target combination to `./output`.

Update `configs/eval.yaml`:

```yaml
model:
  config:
    model_path: ./models/alphaface_demo.pt

data:
  src_path: ./dataset/source
  tar_path: ./dataset/target
  output: ./output
  batch_size: 1
```

Run:

```bash
alphaface-eval
```

or:

```bash
python -m alphaface.eval
```

The shell helper does the same with `CUDA_VISIBLE_DEVICES=0`:

```bash
bash eval.sh
```

### Single Pair Inference

Use `alphaface.infer` for one source/target pair:

```bash
python -m alphaface.infer \
  --source ./examples/source.png \
  --target ./examples/target.png \
  --checkpoint ./models/alphaface_demo.pt \
  --output ./output/swapped.png
```

The source image provides identity. The target image provides pose, expression, lighting, background, and other attributes.

## Export to ONNX

Use the included `export.py` script to export the same inference path used by `alphaface.infer` and `alphaface.eval`: the AlphaFace model called with `(target, source)`.

```bash
python export.py \
  --checkpoint ./models/alphaface_demo.pt \
  --output ./models/alphaface_swapper.onnx \
  --device cuda \
  --opset 17
```

By default, the exported graph has a dynamic batch dimension. Add `--static-batch` if you always serve the same batch size and want a simpler graph. Add `--verify` to run `onnx.checker.check_model` after export.

If export fails because the model uses an operator unsupported by your ONNX opset, try a newer opset supported by your installed PyTorch and ONNX Runtime. If you want a smaller deployment graph, export a lower-level submodule only after checking that its inputs and outputs match the tensors served in production.

## Efficient ONNX Runtime Inference

Install ONNX Runtime:

```bash
pip install onnxruntime-gpu
```

Use `onnxruntime` with CUDA when available and keep preprocessing identical to PyTorch inference:

- Target image: RGB, resized to 256 x 256, float32, channel-first, range `[0, 1]`, shape `[N, 3, 256, 256]`.
- Source image: RGB, resized to 112 x 112, float32, channel-first, normalized to `[-1, 1]` by `(x * 255 / 127.5) - 1`, shape `[N, 3, 112, 112]`.
- Output image: usually `[N, 3, 256, 256]`, convert back to HWC RGB and clamp to `[0, 255]`.

Example:

```python
import cv2
import numpy as np
import onnxruntime as ort


def load_target(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


def load_source(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 127.5 - 1.0
    return np.transpose(x, (2, 0, 1))[None]


providers = [
    ("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kSameAsRequested",
    }),
    "CPUExecutionProvider",
]

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "./models/alphaface_swapper.onnx",
    sess_options=session_options,
    providers=providers,
)

target = load_target("./examples/target.png")
source = load_source("./examples/source.png")

swapped = session.run(
    ["swapped"],
    {
        "target": target,
        "source": source,
    },
)[0]

img = swapped[0].transpose(1, 2, 0)
img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite("./output/swapped_onnx.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

For efficient production inference:

- Create one `InferenceSession` per process and reuse it.
- Batch multiple source/target pairs when latency requirements allow it.
- Keep arrays contiguous `float32` in NCHW layout before calling `session.run`.
- Prefer `onnxruntime-gpu` with `CUDAExecutionProvider` for NVIDIA GPUs.
- Enable full graph optimization with `ORT_ENABLE_ALL`.
- Avoid loading images, creating sessions, or allocating large buffers inside the hot path.
- Consider static batch export if you always serve batch size 1 and want the simplest graph.
- Validate numerical output against PyTorch on a few images before deploying.

## Demo Data

This repository only includes sample-scale data for training and evaluation demo workflows. For meaningful training, prepare a larger aligned face dataset such as VGGFace2-HQ, CelebA-HQ, FF++, MPIE, LPFF, or your own consent-cleared dataset.

## Citation

```bibtex
@article{yu2026alphaface,
  title={AlphaFace: High Fidelity and Real-time Face Swapper Robust to Facial Pose},
  author={Yu, Jongmin and Oh, Hyeontaek and Sun, Zhongtian and Aviles-Rivero, Angelica I and Jeon, Moongu and Yang, Jinhong},
  journal={arXiv preprint arXiv:2601.16429},
  year={2026}
}
```
