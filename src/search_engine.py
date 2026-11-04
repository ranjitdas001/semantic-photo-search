
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class SemanticSearchEngine:

    def __init__(self, embeddings_file=None):

        print("Loading search engine...")

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

        if embeddings_file is None:
            embeddings_file = (
                Path(__file__).parent.parent
                / "image_embeddings.npz"
            )

        data = np.load(embeddings_file)

        self.image_paths = data["image_paths"]
        self.image_embeddings = data["embeddings"]

        print(
            f"Loaded {len(self.image_paths)} image embeddings"
        )

    def generate_text_embedding(self, query):

        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            text_outputs = self.model.text_model(
                **inputs
            )

            pooled_output = text_outputs.pooler_output

            text_features = self.model.text_projection(
                pooled_output
            )

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True
            )

        return text_features.cpu().numpy()[0]

    def search(self, query, top_k=5):

        query_embedding = self.generate_text_embedding(
            query
        )

        # Cosine similarity: embeddings are already normalized
        similarities = (
            self.image_embeddings @ query_embedding
        )

        # Get the indices of the highest scores
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "image_path": self.image_paths[index],
                "score": float(similarities[index])
            })

        return results


if __name__ == "__main__":

    engine = SemanticSearchEngine()

    print("\nSemantic Photo Search")
    print("Type 'exit' to quit")

    while True:

        query = input("\nDescribe the photo: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = engine.search(query, top_k=5)

        print("\nTop matching photos:\n")

        for rank, result in enumerate(results, start=1):

            filename = Path(
                result["image_path"]
            ).name

            score = result["score"]

            print(
                f"{rank}. {filename} "
                f"| Similarity: {score:.4f}"
            )
