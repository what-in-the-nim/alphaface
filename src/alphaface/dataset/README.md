# Dataset Preparation

## Training

Training uses `FaceImageDatasetClip`, which expects a packed dataset:

```text
<db_path>/
└── packed/
    ├── 000001.png
    └── 000002.png
```

Each packed sample is an RGBA PNG:

| Storage location | Content |
| --- | --- |
| RGB channels | Aligned face image. |
| Alpha channel | Binary face mask, with `0` for face and `255` for background. |
| PNG iTXt `alphaface_caption` | Captioner text. |
| PNG iTXt `alphaface_clip_img` | CLIP ViT-B/32 image embedding, 512 float16 values encoded as base64. |
| PNG iTXt `alphaface_clip_txt` | CLIP ViT-B/32 text embedding, 512 float16 values encoded as base64. |
| PNG iTXt `alphaface_id_emb` | ArcFace identity embedding, 512 float16 values encoded as base64. |

The loader fails if `packed/` is missing, empty, or if any PNG is missing required metadata.

Set `db_path` in your config to the dataset root:

```python
config.db_path = "./dataset/custom"
```

## Creating Samples

Use the preparation CLI for raw images:

```bash
alphaface-prepare-dataset \
  --input ./raw_faces \
  --output ./dataset/custom \
  --device cuda \
  --mask-model ./weights/resnet34.onnx \
  --caption-url http://localhost:11434/v1 \
  --caption-model llava:7b \
  --id-encoder-checkpoint ./models/vit_b_fr_pgair.pt
```

Use `--delete-originals` if you want to keep only `packed/` after preparation.

The preparation and conversion commands reuse existing packed metadata by key.
If a packed sample already has a valid CLIP image embedding, that embedding is
kept even when other fields still need to be generated.

Use `--rebuild FIELD` to force one field to be regenerated. Valid fields are
`caption`, `clip_image`, `clip_text`, and `arcface`. Use `--rebuild-all` to
regenerate every metadata field.

## Evaluation

Evaluation still runs pairwise from regular image directories:

```text
<src_path>/    # source identity images
<tar_path>/    # target attribute images
```
