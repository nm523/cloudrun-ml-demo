"""Unit tests for the whole application."""

from unittest.mock import MagicMock

import pytest
from falcon import testing

import app.factory


@pytest.fixture()
def mock_model_session():
    """Create a mock ONNX session for testing."""
    mock_model_session = MagicMock()
    mock_model_session.run.return_value = (None, [{0: 0.0, 1: 0.1, 2: 0.2}])
    return mock_model_session


@pytest.fixture()
def client(mock_model_session):
    """Create the Falcon test client for making API calls."""
    api = app.factory.Factory(mock_model_session).build()
    return testing.TestClient(api)


def test_post_endpoint_returns_expected(client, mock_model_session):
    """Check that the POST endpoint returns the expected data."""
    payload = {
        "sepal_length_cm": 1,
        "sepal_width_cm": 2,
        "petal_length_cm": 3,
        "petal_width_cm": 4,
    }

    expected_output = {
        "predictions": [
            {"class_name": "Iris Setosa", "class_no": 0, "score": 0.0},
            {"class_name": "Iris Versicolour", "class_no": 1, "score": 0.1},
            {"class_name": "Iris Virginica", "class_no": 2, "score": 0.2},
        ]
    }

    result = client.simulate_post("/predict", json=payload)
    assert result.json == expected_output
