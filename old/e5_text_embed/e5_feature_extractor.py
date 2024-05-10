import torch
from sentence_transformers import SentenceTransformer
import time
class E5FeatureExtractor:
    model_name = "intfloat/multilingual-e5-large"
     
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("Device:", self.device)

    def get_passage_features(self, input_texts, verbose=False):
        texts = ["passage: " + text for text in input_texts]
        return self.embed_texts(texts, verbose=verbose)

    def get_query_features(self, input_texts, verbose=False):
        texts = ["query: " + text for text in input_texts]
        return self.embed_texts(texts, verbose=verbose)
    
    def embed_texts(self, texts, verbose=False):

        start_time = time.perf_counter()
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        end_time = time.perf_counter()

        avg_len = sum([len(ele) for ele in texts]) / len(texts)
        if verbose:
            print(f"E5: Batch size: {len(texts)}, Average Length: {avg_len}, Time: {end_time - start_time}s")
        return embeddings