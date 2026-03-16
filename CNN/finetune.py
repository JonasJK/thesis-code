import logging
import os
import sys

import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from CNN.cnn_nir import CNNNir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MODEL_PATH = "best_cnn_model.pth"
RGBI_FILE = "/work/klugej/clc/312/312.tif"
EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZE = 32


def main():
    log.info("Starting finetuning process")
    log.info(f"  Model: {MODEL_PATH}")
    log.info(f"  RGBI file: {RGBI_FILE}")
    log.info(f"  Epochs: {EPOCHS}")
    log.info(f"  Learning rate: {LEARNING_RATE}")
    log.info(f"  Batch size: {BATCH_SIZE}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(RGBI_FILE):
        raise FileNotFoundError(f"RGBI file not found: {RGBI_FILE}")

    try:
        log.info("Initializing CNN model...")
        model = CNNNir(
            patch_size=50,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            validation_split=0.15,
        )

        log.info(f"Loading pre-trained model from {MODEL_PATH}...")
        model.load_model(MODEL_PATH)
        log.info("Model loaded")

        log.info(f"Loading RGBI file: {RGBI_FILE}")
        rgb_image, nir_image = model.load_rgbi_image(RGBI_FILE)
        log.info(f"Image loaded - RGB shape: {rgb_image.shape}, NIR shape: {nir_image.shape}")

        log.info("Extracting patches...")
        patches_rgb, patches_nir = model._extract_patches(rgb_image, nir_image)
        log.info(f"Extracted {len(patches_rgb)} patches")

        model.patches_rgb = patches_rgb
        model.patches_nir = patches_nir

        log.info("Starting finetuning")
        model.fit_model()

        output_path = MODEL_PATH.replace(".pth", "_finetuned.pth")
        log.info(f"Saving finetuned model to {output_path}...")
        torch.save(model.model.state_dict(), output_path)

        log.info("=" * 60)
        log.info("FINETUNING COMPLETE")
        log.info(f"Finetuned model saved to: {output_path}")
        log.info("=" * 60)

    except Exception as e:
        log.error(f"Finetuning failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
