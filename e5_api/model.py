from PIL import Image
from inference_api import Model

import torch
from sentence_transformers import SentenceTransformer
import time

class E5(Model):
    model_name = "intfloat/multilingual-e5-large"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("Device:", self.device)


    def predict(self, embed_type, texts):
        if embed_type == "query":
            texts = ["query: " + text for text in texts]
        elif embed_type == "passage":
            texts = ["passage: " + text for text in texts]

        feature_vectors = self.model.encode(texts, normalize_embeddings=True)
        return feature_vectors
    
