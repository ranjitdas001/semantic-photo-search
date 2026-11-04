
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


# Dataset location
DATASET_PATH = Path(
    r"D:\Downloads\archive\data.pickle"
)

# Output folder
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "photos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_images():
    print("Loading dataset...")

    with open(DATASET_PATH, "rb") as file:
        dataset = pickle.load(file, encoding="bytes")

    print("Dataset keys:", dataset.keys())

    images = dataset[b"data"]

    # Optional labels
    labels = dataset.get(b"labels", None)

    print("Total images:", len(images))

    for i, image_data in enumerate(images):

        # CIFAR-10: RGB channels stored separately
        image = image_data.reshape(3, 32, 32)

        # Convert to H x W x C format
        image = image.transpose(1, 2, 0)

        image = Image.fromarray(image)

        # Save image
        if labels is not None:
            filename = f"{labels[i]}_{i}.png"
        else:
            filename = f"image_{i}.png"

        image.save(OUTPUT_DIR / filename)

        if (i + 1) % 1000 == 0:
            print(f"Extracted {i + 1} images")

    print("\nExtraction completed!")
    print("Images saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    extract_images()
