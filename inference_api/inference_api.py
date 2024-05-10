# First-party
import os
from pathlib import Path
import logging
from typing import List, Callable
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
import asyncio

# Second-party
from .process_functions import worker_create_model, worker_model_predict
from .model import Model

# Third-party
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

class InferenceAPI(FastAPI):
    pool: ProcessPoolExecutor
    logger: logging.Logger
    warmup: Callable

    def __init__(self, 
            model_type: Model, 
            warmup: Callable = None,
            workers=1, 
            **kwargs
        ):
        super().__init__( lifespan=self.lifespan, docs_url=None, redoc_url=None, **kwargs)
        self.warmup = warmup
        self.logger = logging.getLogger('uvicorn.error')
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_create_model,
            initargs=(model_type,)
        )
        static_directory = Path(__file__).parent / "static"
        self.mount('/static', StaticFiles(directory=static_directory), name="static")
        self.add_api_route("/docs", self.docs, methods=["GET"], include_in_schema=False)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Warm up model 
        if self.warmup is not None:
            self.logger.info("Running model in worker pool as warmup")
            await self.warmup()

        # Let FastAPI take over
        self.logger.info("Starting API")
        yield
        # After FastAPI end
        self.logger.info("API shutdown")

        # Shutdown worker pool
        self.logger.info("Shutting down worker pool")
        self.pool.shutdown()
 
    def docs(self):
        return get_swagger_ui_html(
            openapi_url=self.openapi_url,
            title=self.title,
            swagger_favicon_url=f'/static/favicon.png',
            swagger_js_url=f'/static/swagger-ui-bundle.js',
            swagger_css_url=f'/static/swagger-ui.css'
        )
    
    async def warmup():
        pass

    # Wrap the model prediction in the worker as a Future (awaitable)
    async def predict(self, *args):
        loop = asyncio.get_running_loop()
        inference_time, result = await loop.run_in_executor(self.pool, worker_model_predict, args)
        return inference_time, result