"""CLI for the intermediate flat visible-symbol sequence stage."""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from formula_symbol_training.config import FormulaSymbolConfig
    from formula_symbol_training.data import create_dataloaders
    from formula_symbol_training.prepare_data import prepare_data
    from formula_symbol_training.trainer import Trainer
else:
    from .config import FormulaSymbolConfig
    from .data import create_dataloaders
    from .prepare_data import prepare_data
    from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the flat formula-symbol stage.")
    parser.add_argument("--epochs", type=int, default=FormulaSymbolConfig.NUM_EPOCHS)
    parser.add_argument("--patience", type=int,
                        default=FormulaSymbolConfig.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--preview-every", type=int,
                        default=FormulaSymbolConfig.PREVIEW_EVERY)
    parser.add_argument("--init-checkpoint", type=Path,
                        default=FormulaSymbolConfig.INIT_CHECKPOINT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--from-scratch", action="store_true",
                        help="Do not initialize from the patch-2 single-symbol checkpoint.")
    args = parser.parse_args()
    if args.resume and args.from_scratch:
        parser.error("--resume cannot be combined with --from-scratch.")
    if args.resume and args.init_checkpoint != FormulaSymbolConfig.INIT_CHECKPOINT:
        parser.error("Use --resume or --init-checkpoint, not both.")

    data_dir = prepare_data()
    config = FormulaSymbolConfig()
    config.FLAT_DATA_DIR = data_dir
    train_loader, val_loader = create_dataloaders(config)
    vocab = train_loader.dataset.vocab
    init_checkpoint = None if args.from_scratch or args.resume else args.init_checkpoint
    if init_checkpoint and not init_checkpoint.is_file():
        raise FileNotFoundError(
            f"Patch-2 single-symbol checkpoint not found: {init_checkpoint}. "
            "Train single_symbol_training first, or pass --from-scratch."
        )

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        epochs=args.epochs,
        patience=args.patience,
        preview_every=args.preview_every,
        resume_path=args.resume,
        init_checkpoint=init_checkpoint,
        from_scratch=args.from_scratch,
    )
    trainer.train()


if __name__ == "__main__":
    main()
