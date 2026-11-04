import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# Set page configuration
st.set_page_config(page_title="Semantic Photo Search", layout="wide")
st.title("Semantic Photo Search 🔍")
st.markdown("Search your local gallery using natural language.")

# 1. Load Model and Processor (Cached to prevent reloading on every search)
@st.cache_resource
def load_clip_model():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

# 2. Load Embeddings (Cached for speed)
@st.cache_data
def load_embeddings(filepath="image_embeddings.npz"):
    if not os.path.exists(filepath):
        return None, None
    data = np.load(filepath)
    # Note: If your image_indexer.py used different keys inside np.savez(), change 'embeddings' and 'paths' here.
    # return torch.tensor(data['embeddings']), data['paths']
    return torch.tensor(data['embeddings']), data['image_paths']

with st.spinner("Loading CLIP model and embeddings..."):
    model, processor = load_clip_model()
    image_embeddings, image_paths = load_embeddings("image_embeddings.npz")

if image_embeddings is None:
    st.error("Could not find `image_embeddings.npz`. Make sure you have run your indexer first.")
    st.stop()

# 3. Search Interface
query = st.text_input("Enter a description (e.g., 'a girl holding a phone'):")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

if query:
    with st.spinner("Searching..."):
        
        # 1. Intent Detection Logic
        def detect_intent(user_input):
            text = user_input.lower().strip()
            
            if text.startswith(("a ", "an ", "the ", "photo of", "image of", "picture of", "screenshot of")):
                return text
                
            if any(word in text for word in ["screenshot", "code", "app", "ui", "chat", "message"]):
                return f"a digital screenshot of {text}"
            elif any(word in text for word in ["receipt", "document", "text", "bill", "letter"]):
                return f"a scanned document showing {text}"
            elif any(word in text for word in ["meme", "drawing", "art", "sketch", "logo", "icon"]):
                return f"a graphic or illustration of {text}"
            elif any(word in text for word in ["selfie", "face", "portrait"]):
                return f"a close-up portrait of {text}"
            else:
                return f"a high-quality photo of {text}"

        smart_query = detect_intent(query)
        st.caption(f"*(AI interpreted your intent as: '{smart_query}')*")
        
        # 2. Bulletproof feature extraction using a dummy image
        # This guarantees the text goes through the final multimodal projection layer
        dummy_image = Image.new('RGB', (224, 224), color='black')
        inputs = processor(text=[smart_query], images=dummy_image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Extracted after all projections are complete
            text_features = outputs.text_embeds
        
        # Normalize features for cosine similarity
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        img_features = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarities = torch.matmul(text_features, img_features.T).squeeze(0)
        
        # Get top K results
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        # 4. Display Results
        st.subheader(f"Top {top_k} results for: '{query}'")
        
        cols = st.columns(top_k)
        
        for idx, col in enumerate(cols):
            img_idx = top_indices[idx].item()
            img_path = str(image_paths[img_idx])
            score = top_scores[idx].item()
            
            with col:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    st.caption(f"Score: {score:.3f}")
                except Exception:
                    st.error(f"Error loading image: {os.path.basename(img_path)}")