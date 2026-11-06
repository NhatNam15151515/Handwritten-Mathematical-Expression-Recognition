# Flat formula-symbol training

This intermediate source trains the shared patch-2 ViT and a sequence decoder on CROHME
formula images. It removes position/grouping syntax from derived captions so this stage
focuses on reading visible glyphs in a flat sequence. Original CROHME captions and images
remain untouched; only derived caption files are written to
`formula_symbol_training/data/crohme_whywhs_source/flat_symbol_sequences/`.

This run uses a maximum `44x44` feature grid (`MAX_PATCH_GRID`). Its checkpoints, logs,
and validation predictions are stored under the `grid44` run name.

Target conversion rules are in `target_format.py`:

- Remove `{`, `}`, `^`, `_`, and `\\limits` because they encode grouping or placement.
- Keep escaped braces `\\{`, `\\}` and visible square brackets; remove square-bracket
  delimiters only when they encode an optional radical index.
- Map `\\frac` to `<FRAC_BAR>` and `\\sqrt` to `<RADICAL>`.
- Expand handwritten function names such as `\\sin` and `\\cos` into their letters.

Start after the processed patch-2 single-symbol classifier has saved its best checkpoint:

```powershell
.\.venv\Scripts\python.exe -m formula_symbol_training.train --epochs 150 --patience 15
```

Token-level validation loss is calculated each epoch. Every 5 epochs, the source also
calculates greedy BLEU/exact-match metrics and writes a CSV containing predictions for every
validation image under
`formula_symbol_training/outputs/vit_patch2_formula_symbols_grid44/validation_predictions/`.
Early stopping and `best_generation.pth` follow generated BLEU-4, with exact match as a
tie-breaker. `best.pth` continues to track the lowest teacher-forced validation loss.

The initialized encoder checkpoint is
`single_symbol_training/checkpoints/vit_patch2_single_symbol_classifier_processed/best.pth`.
The source saves its own checkpoints in
`formula_symbol_training/checkpoints/vit_patch2_formula_symbols_grid44/`.
Resume the interrupted epoch-50 run with the revised 150-epoch schedule:

```powershell
.\.venv\Scripts\python.exe -m formula_symbol_training.train --epochs 150 --patience 15 --preview-every 5 --resume formula_symbol_training\checkpoints\vit_patch2_formula_symbols_grid44\latest.pth
```

Use `best_generation.pth` when selecting a checkpoint for generated sequence quality;
use `best.pth` for minimum teacher-forced validation loss.
For the later full-formula stage, transfer `vision_encoder_state_dict`; its flat-sequence
decoder vocabulary intentionally differs from the original formula decoder vocabulary.
