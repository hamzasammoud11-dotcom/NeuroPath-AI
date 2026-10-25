# OASIS-1 CNN v003 inference

This is a new, inference-only directory. It does not alter the OASIS-1 dataset, model sources, checkpoints, study records, or held-out patients. The module accepts one subject's nine original-view slices, applies the frozen 128x128 grayscale bilinear preprocessing and `(x - 0.5) / 0.5` normalization, computes positive-class probabilities, and returns their arithmetic mean with the protocol threshold `0.5`.

The loader is fail-closed: the file basename must be `neuropath-oasis1-cnn-v003.pt`, its SHA-256 must equal `349ccc59ec3ba8dc834fdca7f426eb50fc8d1e5be38cb8d0bd58ab7a30c73e60`, and its v003 payload must identify `model_state_dict`, the selected `horizontal_flip` variant, epoch `18`, and the frozen aggregation/threshold. Generic `image2d_v1` preprocessing/class-mapping metadata is also checked when present. The checkpoint is not included in this repository; place the exact authorized file at a private path and do not substitute another checkpoint.

From this directory, install the project dependencies, then run the synthetic tests with:

```text
python -m unittest -v test_inference.py
```

Run one subject using exactly nine paths:

```text
python cli.py --checkpoint C:\private\neuropath-oasis1-cnn-v003.pt --subject-id OAS1_XXXX --slice s052.png --slice s061.png --slice s070.png --slice s079.png --slice s088.png --slice s096.png --slice s105.png --slice s114.png --slice s122.png
```

The CLI prints JSON containing the nine slice probabilities, `cdr_score`, `prediction` (`0` or `1`), and threshold. Use only original-view slices from one subject; do not pass the 40 held-out test patients. The result is exploratory CDR-associated classification and carries no clinical or anatomical-laterality claim.
