
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageIndexer:

    def __init__(self):
        print("Loading CLIP model...")

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_name = "openai/clip-vit-base-patch32"

        self.model = CLIPModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )

        self.model.eval()

        print(f"Model loaded on {self.device}")

    # IMPORTANT: This method must be inside the class
    def generate_embedding(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(
            self.device
        )

        with torch.no_grad():

            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values
            )

            pooled_output = vision_outputs.pooler_output

            image_features = self.model.visual_projection(
                pooled_output
            )

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )

        return image_features.cpu().numpy()[0]

    # IMPORTANT: This method must also be inside the class

    def index_folder(self, folder_path):

        folder = Path(folder_path)

        image_paths = []
        embeddings = []

        supported_formats = {
            ".jpg", ".jpeg", ".png", ".webp"
        }

        image_files = sorted([
            file for file in folder.iterdir()
            if file.suffix.lower() in supported_formats
        ])

        print(f"Found {len(image_files)} images")

        for image_path in image_files:

            try:
                embedding = self.generate_embedding(image_path)

                image_paths.append(str(image_path))
                embeddings.append(embedding)

                print(f"Indexed: {image_path.name}")

            except Exception as error:
                print(f"Failed: {image_path.name} - {error}")

        return image_paths, embeddings


if __name__ == "__main__":

    indexer = ImageIndexer()

    folder = (
        Path(__file__).parent.parent
        / "data"
        / "coco_val2017"
        / "val2017"
    )

    image_paths, embeddings = indexer.index_folder(folder)

    if embeddings:

        np.savez(
            "image_embeddings.npz",
            image_paths=np.array(image_paths),
            embeddings=np.array(embeddings)
        )

        print("\nIndexing completed!")
        print(f"Total images indexed: {len(image_paths)}")

