"""Machine learning model resource.

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
"""
from typing import Any, Dict, List, Union

import falcon
import numpy as np
from falcon.media.validators import jsonschema
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

IRIS_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "sepal_length_cm": {"type": "number"},
        "sepal_width_cm": {"type": "number"},
        "petal_length_cm": {"type": "number"},
        "petal_width_cm": {"type": "number"},
    },
}
"""Dict: Input Schema for the Iris model."""

IRIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "properties": {
                "class_no": {"type": "integer"},
                "class_name": {"type": "string"},
                "score": {"type": "number"},
            },
        }
    },
}
"""Dict: Output Schema for the Iris model."""

IRIS_CLASSES = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
"""Dict[int, str]: Map model classes to their outputs."""

IRIS_INPUT_FEATURES = list(IRIS_REQUEST_SCHEMA["properties"])
"""List[str]: Ordered list of input features for the model."""


class IrisResource:
    """The API resource for the model."""

    def __init__(self, model_session: InferenceSession):
        """Initiate the resource by injecting the model as a dependency."""
        self.model_session = model_session
        self.input_name = model_session.get_inputs()[0].name

    @jsonschema.validate(
        req_schema=IRIS_REQUEST_SCHEMA, resp_schema=IRIS_RESPONSE_SCHEMA
    )
    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """For an input."""
        resp.media = {"predictions": self._predict_and_marshall(req.media)}
        resp.status = falcon.HTTP_OK

    @staticmethod
    def _convert_dict_to_ordered_array(
        input_dict: Dict[Any, Any], keys: List[str]
    ) -> np.array:
        """Converts the input dictionary to an ordered array before passing to the ONNX session.

        Args:
            input_dict: The input dictionary to convert.
            schema: The keys in order they should appear in the array
        """
        return np.array([[input_dict[key] for key in keys]], dtype=np.float32)

    @staticmethod
    def _marshall_predictions(
        prediction: Dict[int, str]
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Format the predictions into a list of results for the client to receive."""
        return [
            {"class_no": class_no, "class_name": IRIS_CLASSES[class_no], "score": score}
            for class_no, score in prediction.items()
        ]

    def _predict(self, dimensions: Dict[str, float]) -> Dict[str, Any]:
        """Make a prediction for one input only."""
        # Make sure the inputs are going in in the correct order.
        model_inputs = self._convert_dict_to_ordered_array(
            dimensions, IRIS_INPUT_FEATURES
        )

        # Select the raw scores (probabilities) prior to formatting them.
        return self.model_session.run(None, {self.input_name: model_inputs})[1][0]

    def _predict_and_marshall(self, dimensions: Dict[str, float]) -> Dict[str, Any]:
        """Predicts the result and formats it for returning to the client."""
        prediction = self._predict(dimensions)
        return self._marshall_predictions(prediction)
