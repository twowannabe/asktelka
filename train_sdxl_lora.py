#!/usr/bin/env python3
"""
Train SDXL LoRA on Replicate for SWEENY_FACE identity.

Usage:
    python3 train_sdxl_lora.py

Requires:
    - REPLICATE_API_TOKEN env var
    - Archive.zip with training photos in repo root
    - replicate Python package: pip install replicate

The script:
1. Uploads Archive.zip to Replicate file hosting
2. Creates SDXL LoRA training with optimized params for face identity
3. Polls until training completes
4. Prints the new model version to use in config.py
"""

import os
import sys
import time

try:
    import replicate
except ImportError:
    print("Install replicate: pip install replicate")
    sys.exit(1)

# --- Config ---
ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "Archive.zip")
# Destination model on Replicate (your_username/model_name)
DESTINATION = os.environ.get("REPLICATE_DESTINATION", "twowannabe/sweeny-sdxl")

TRAINING_PARAMS = {
    "input_images": None,  # will be set after upload
    "token_string": "SWEENY_FACE",
    "caption_prefix": "a photo of SWEENY_FACE, ",
    "resolution": 1024,
    "max_train_steps": 1500,
    "train_batch_size": 2,
    "unet_learning_rate": 1e-6,
    "lora_lr": 2e-4,
    "ti_lr": 5e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "crop_based_on_salience": True,
    "use_face_detection_instead": True,
    "is_lora": True,
    "verbose": True,
    "seed": 42,
}


def main():
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Error: set REPLICATE_API_TOKEN environment variable")
        sys.exit(1)

    if not os.path.exists(ARCHIVE_PATH):
        print(f"Error: {ARCHIVE_PATH} not found")
        sys.exit(1)

    print(f"Uploading {ARCHIVE_PATH} to Replicate...")
    with open(ARCHIVE_PATH, "rb") as f:
        file_output = replicate.files.create(f, filename="Archive.zip", content_type="application/zip")
    file_url = file_output.urls["get"]
    print(f"Uploaded: {file_url}")

    TRAINING_PARAMS["input_images"] = file_url
    TRAINING_PARAMS["input_images_filetype"] = "zip"

    # Create destination model if it doesn't exist
    owner, model_name = DESTINATION.split("/")
    try:
        replicate.models.get(DESTINATION)
        print(f"Model {DESTINATION} already exists")
    except Exception:
        print(f"Creating model {DESTINATION}...")
        replicate.models.create(
            owner=owner,
            name=model_name,
            visibility="private",
            hardware="gpu-t4",
        )
        print(f"Model created: {DESTINATION}")

    print(f"\nStarting SDXL LoRA training → {DESTINATION}")
    print(f"Token: {TRAINING_PARAMS['token_string']}")
    print(f"Steps: {TRAINING_PARAMS['max_train_steps']}")
    print(f"Resolution: {TRAINING_PARAMS['resolution']}")
    print()

    training = replicate.trainings.create(
        version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input=TRAINING_PARAMS,
        destination=DESTINATION,
    )

    print(f"Training started: {training.id}")
    print(f"Status URL: https://replicate.com/p/{training.id}")
    print("\nPolling for completion...\n")

    while training.status in ("starting", "processing"):
        time.sleep(15)
        training.reload()
        elapsed = ""
        metrics = getattr(training, "metrics", None)
        if metrics and "predict_time" in metrics:
            elapsed = f" ({metrics['predict_time']:.0f}s)"
        print(f"  Status: {training.status}{elapsed}")

    if training.status == "succeeded":
        print(f"\n✅ Training succeeded!")
        print(f"\nOutput: {training.output}")
        print(f"\n--- UPDATE config.py ---")
        print(f'SELFIE_LORA_MODEL = "{DESTINATION}:<version_hash>"')
        print(f"\nCheck your model at: https://replicate.com/{DESTINATION}")
        print("Copy the version hash from the model page and update config.py")
    else:
        print(f"\n❌ Training failed: {training.status}")
        if training.error:
            print(f"Error: {training.error}")
        if training.logs:
            print(f"\nLogs (last 500 chars):\n{training.logs[-500:]}")


if __name__ == "__main__":
    main()
