import sys, os
sys.path.insert(0, os.path.abspath('..'))

from typing import List

from fastapi import File, UploadFile
from pydantic import BaseModel, Field

from model import E5
from inference_api import InferenceAPI

class TextInputModel(BaseModel):
    texts: List[str] = Field(examples=[["Today is a day. Go catch them all!"]])
 
class OutputModel(BaseModel):
    feature_vectors: List[List[float]]
    inference_time: float

async def warmup():
    await app.predict("query", ["Today is a day. Go catch them all!"])

title = "E5 Multilingual Model Service"
description = """
This a description of the E5 Multilingual model
"""
app = InferenceAPI(model_type=E5, title=title, description=description, warmup=warmup)

@app.post('/query')
async def texts_embedding(input: TextInputModel) -> OutputModel:
    inference_time, feature_vectors = await app.predict("query", input.texts)
    return OutputModel(
        feature_vectors = feature_vectors,
        inference_time = inference_time,
    )

@app.post('/passage')
async def images_embedding(images: List[UploadFile] = File(...)) -> OutputModel:
    inference_time, feature_vectors = await app.predict("passage", input.texts)
    return OutputModel(
        feature_vectors = feature_vectors,
        inference_time = inference_time,
    )