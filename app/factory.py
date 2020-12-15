"""Falcon application builder.

This is used to construct and install dependencies for the application.
For example, this could be configuring and registering middleware.
"""
import falcon
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

from .iris import IrisResource


class Factory:
    """Application Factory."""

    def __init__(self, model_session: InferenceSession):
        self.api = falcon.API()
        self.model_session = model_session

    def _register_middleware(self) -> None:
        """Registers middleware like SQL database connections."""
        pass

    def _register_routes(self) -> None:
        """Registers resources against routes for the application."""
        iris_resource = IrisResource(self.model_session)
        self.api.add_route("/predict", iris_resource)

    def build(self) -> falcon.API:
        """Builds and returns the application."""
        self._register_middleware()
        self._register_routes()
        return self.api
