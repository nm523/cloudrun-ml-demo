"""Main entrypoint for Gunicorn."""

import os

import numpy
import onnxruntime as rt

from .builder import Factory

# Load the ONNX model prior to injecting into the API.
model_path = os.environ["IRIS_ONNX_MODEL_PATH"]
model_session = rt.InferenceSession(model_path)

# Instantiate the API object.
api = Factory(model_session=model_session).build()
