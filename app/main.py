"""Main entrypoint for Gunicorn."""

import os

import onnxruntime as rt

from .factory import Factory

# Load the ONNX model prior to injecting into the API.
model_path = os.environ["IRIS_ONNX_MODEL_PATH"]
model_session = rt.InferenceSession(model_path)

api = Factory(model_session=model_session).build()
