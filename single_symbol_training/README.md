# Single-symbol training source

This package trains a standalone patch-2 ViT classifier on
`single_symbol_training/data/single_symbol_native/`.
For this pretraining run, `C/P/S/V/X/Y/o` are mapped to `c/p/s/v/x/y/0` in the processed caption
set at `single_symbol_training/data/single_symbol_processed/`. The original image arrays are reused without
copying, and formula-stage labels are not changed. The training command prepares the
derived caption set automatically.
It shares the repository's image preprocessing and ViT encoder implementation while keeping
the single-symbol classifier, training loop, reports, and checkpoints separate from formula training.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m single_symbol_training.train --epochs 1000 --preview-every 5
```

Outputs:

- Best and latest checkpoints: `single_symbol_training/checkpoints/vit_patch2_single_symbol_classifier_processed/`
- Per-epoch history and class error analysis: `logs/vit_patch2_single_symbol_classifier_processed/`
- Per-class validation prediction boards and full per-image prediction CSVs (every 5 epochs by default): `single_symbol_training/outputs/vit_patch2_single_symbol_classifier_processed/`

Start a new processed-label run using the existing model's encoder weights but a fresh classifier head:

```powershell
.\.venv\Scripts\python.exe -m single_symbol_training.train --epochs 1000 --init-encoder single_symbol_training\checkpoints\vit_patch2_single_classifier\latest.pth
```

Resume that processed-label run after interruption:

```powershell
.\.venv\Scripts\python.exe -m single_symbol_training.train --epochs 1000 --resume single_symbol_training\checkpoints\vit_patch2_single_symbol_classifier_processed\latest.pth
```

The saved checkpoint includes `vision_encoder_state_dict` for later transfer into a formula model
that also uses patch size 2. Its classifier head is specific to the single-symbol classes.
