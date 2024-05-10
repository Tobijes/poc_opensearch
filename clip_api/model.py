from PIL import Image
from inference_api import Model

import torch
from transformers import CLIPModel, CLIPProcessor

class CLIP(Model):
    model_name = "openai/clip-vit-base-patch32"


    def __init__(self):
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)
        self.model.to(self.device)


    def predict(self, input_type, texts_or_paths):
        if input_type == "text":
            feature_vectors = self.get_text_features(texts_or_paths)
        elif input_type == "image":
            images = [Image.open(image_path).convert("RGB") for image_path in texts_or_paths]
            feature_vectors = self.get_image_features(images)
        else:
            feature_vectors = []

        return feature_vectors
    

    @torch.no_grad()
    def get_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = inputs.to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.tolist()
        return text_features


    @torch.no_grad()
    def get_image_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.tolist()
        return image_features


