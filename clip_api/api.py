import sys, os
sys.path.insert(0, os.path.abspath('..'))

from typing import List
import tempfile
import os

import aiofiles
from fastapi import File, UploadFile
from pydantic import BaseModel, Field

from model import CLIP
from inference_api import InferenceAPI

class TextInputModel(BaseModel):
    texts: List[str] = Field(examples=[["Today is a day. Go catch them all!"]])

class ImageInputModel(BaseModel):
    images: List[UploadFile]
 
class OutputModel(BaseModel):
    feature_vectors: List[List[float]]
    inference_time: float

async def warmup():
    await app.predict("text", ["Today is a day. Go catch them all!"])

title = "CLIP Model Service"
description = """
This a description of the CLIP model
"""

app = InferenceAPI(model_type=CLIP, title=title, description=description, warmup=warmup)

@app.post('/texts')
async def texts_embedding(input: TextInputModel) -> OutputModel:
    inference_time, feature_vectors = await app.predict("text", input.texts)
    return OutputModel(
        feature_vectors = feature_vectors,
        inference_time = inference_time,
    )

@app.post('/images')
async def images_embedding(images: List[UploadFile] = File(...)) -> OutputModel:
    image_paths = []
    # Save images to disk, to be loaded by other process
    for image in images:
        file = tempfile.NamedTemporaryFile(delete=False)
        image_path = file.name
        file.close()
        image_paths.append(image_path)
        async with aiofiles.open(image_path, 'wb') as out_file:
            while content := await image.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk'

    try:
        inference_time, feature_vectors = await app.predict("image", image_paths)
    except Exception as e:
        app.logger.error(f"Prediction exception: {e}")
    finally:
        # Cleanup stored images
        for image_path in image_paths:
            os.remove(image_path)

    return OutputModel(
        feature_vectors = feature_vectors,
        inference_time = inference_time,
    )