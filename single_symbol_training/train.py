"""Command-line entry point for patch-2 single-symbol training."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    # Support `python single_symbol_training/train.py` as well as `python -m ...`.
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from single_symbol_training.config import SingleSymbolConfig
    from single_symbol_training.data import create_dataloaders
    from single_symbol_training.prepare_processed_data import prepare_processed_data
    from single_symbol_training.trainer import Trainer
else:
    from .config import SingleSymbolConfig
    from .data import create_dataloaders
    from .prepare_processed_data import prepare_processed_data
    from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the patch-2 single-symbol ViT classifier.")
    parser.add_argument("--epochs", type=int, default=SingleSymbolConfig.EPOCHS)
    parser.add_argument("--patience", type=int, default=SingleSymbolConfig.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--preview-every", type=int, default=SingleSymbolConfig.PREVIEW_EVERY,
                        help="Save a validation prediction board every N epochs; 0 disables previews.")
    parser.add_argument("--lr", type=float, default=SingleSymbolConfig.LEARNING_RATE)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from this patch-2 classifier checkpoint.")
    parser.add_argument("--init-encoder", type=Path, default=None,
                        help="Initialize only the ViT encoder from a compatible checkpoint; "
                             "the classifier head and optimizer start fresh.")
    args = parser.parse_args()
    if args.resume and args.init_encoder:
        parser.error("Use either --resume or --init-encoder, not both.")

    config = SingleSymbolConfig()
    manifest = prepare_processed_data(config)
    print(f"Prepared processed single-symbol labels: {manifest['derived_caption_dir']}")
    train_loader, val_loader, class_names = create_dataloaders(config)
    val_dataset = val_loader.dataset
    print(f"Loaded {len(train_loader.sampler)} balanced train samples per epoch, "
          f"{len(val_dataset)} validation images, {len(class_names)} classes.")

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        class_names=class_names,
        epochs=args.epochs,
        patience=args.patience,
        preview_every=args.preview_every,
        learning_rate=args.lr,
        resume_path=args.resume,
        init_encoder_path=args.init_encoder,
    )
    trainer.train()


if __name__ == "__main__":
    main()
