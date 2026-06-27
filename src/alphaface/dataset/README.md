# Dataset Preparation

## Training

Training uses `FaceImageDataset_CLIP` (CLIP-guided) or `FaceImageDataset` (standard).
Both loaders expect a single root directory with three sub-folders:

```
<db_path>/
├── img/        # aligned face images  (PNG or JPG, 256×256)
├── mask/       # binary face masks    (same filenames as img/)
└── txt/        # CLIP text captions   (same base filename, .txt extension)
```

Each sample's three files must share the same stem:

```
img/000002.png   ←→   mask/000002.png   ←→   txt/000002.txt
```

The `txt/` directory is only required when using `FaceImageDataset_CLIP` /
`get_dataloader_clip`. `FaceImageDataset` and `get_dataloader` only need `img/`
and `mask/`.

Set `db_path` in your config to point at this directory:

```python
config.db_path = "./dataset/ffhq_lpff/256_small"
```

### Image requirements

| Property   | Value |
|------------|-------|
| Resolution | 256×256 |
| Format     | PNG or JPG |
| Alignment  | Face-centred (FFHQ-style crop) |

Masks are **inverted** by the loader (`1 − mask`), so a white (255) pixel in the
mask file means *background*, and black (0) means *face*. Generate masks so that
the face region is black and non-face is white, or flip your convention here.

### FFHQ + LPFF (recommended)

The sample data in `ffhq_lpff/256_small/` is sourced from
[FFHQ](https://github.com/NVlabs/ffhq-dataset) and pre-processed with
[LPFF](https://github.com/JD-hwang/LPFF-dataset) for large-pose alignment.

1. Download FFHQ 256×256 images.
2. Run face alignment and crop to 256×256.
3. Generate face segmentation masks (e.g. with
   [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)).
   Binarise: face pixels → 0, background → 255.
4. Generate per-image CLIP captions and save each as `<stem>.txt` with one
   caption per file (the loader reads the first non-empty line).

### Custom dataset

Any aligned face dataset works. Minimum steps:

```bash
# 1. Resize images to 256×256 and put them in img/
mkdir -p dataset/custom/img dataset/custom/mask dataset/custom/txt

# 2. Generate masks (example using face-parsing.PyTorch output)
#    Save binary masks with matching filenames to mask/

# 3. Generate CLIP captions (optional, only needed for CLIP training)
#    One caption per .txt file, matching the image stem
```

Then set `config.db_path = "./dataset/custom"`.

---

## Evaluation

Evaluation runs pairwise: every source identity image is swapped onto every
target image. Two flat directories are required:

```
<src_path>/    # source identity images  (any resolution, resized to 112×112)
<tar_path>/    # target attribute images (any resolution, resized to 256×256)
```

Set in the eval config:

```python
config.src_path = "./dataset/source"
config.tar_path = "./dataset/target"
```

Output filenames are `<source_stem>_<target_filename>` saved to `config.output`.

---

## Directory layout reference

```
dataset/
├── ffhq_lpff/
│   └── 256_small/          # sample training data (subset)
│       ├── img/
│       ├── mask/
│       └── txt/
├── source/                 # sample eval source identities
└── target/                 # sample eval target images
```

---

## `TrainFaceDataSet` (legacy loader)

`data_loader.py` contains an older `TrainFaceDataSet` that expects a different
layout: alongside each image folder it looks for sibling directories named
`<folder>_lm_images/` (landmark renders) and `<folder>_mask_images/` (grayscale
masks). This loader is not used by the default training script (`train_clip.py`)
but is kept for reference.
