"""Tests for the formatting of model outputs."""

from unittest.mock import MagicMock

import numpy as np

from app.iris import IrisResource


def test_marshalling_predictions():
    """Check that the marshalling function produces the correct output."""
    resource = IrisResource(MagicMock())
    predictions = {0: 1, 1: 0.5, 2: 1.5}
    marshalled_results = resource._marshal_predictions(predictions)
    assert marshalled_results == [
        {"class_no": 0, "class_name": "Iris Setosa", "score": 1},
        {"class_no": 1, "class_name": "Iris Versicolour", "score": 0.5},
        {"class_no": 2, "class_name": "Iris Virginica", "score": 1.5},
    ]


def test_ordered_dict_conversion():
    """Check that dictionaries can be marshalled into the correct order."""
    resource = IrisResource(MagicMock())
    input_dict = {"a": 1, "b": 2, "c": 3}
    key_order = ["c", "b", "a"]
    output_array = resource._convert_dict_to_ordered_array(input_dict, key_order)
    expected_array = np.array([[3, 2, 1]], dtype=np.float32)
    assert np.array_equal(output_array, expected_array)


def test_model_session_called_with_correct_inputs():
    """Test that calling the predict method makes the expected call to the model session."""
    mock_session = MagicMock()
    resource = IrisResource(mock_session)

    # Override the input name for the test.
    resource.input_name = "test_name"
    payload = {
        "sepal_length_cm": 1,
        "sepal_width_cm": 2,
        "petal_length_cm": 3,
        "petal_width_cm": 4,
    }
    resource._predict(payload)

    # Inspect the call args directly so we can check with numpy.
    called_with_array = mock_session.run.call_args_list[0][0][1]["test_name"]
    expected_array = np.array([[1, 2, 3, 4]], dtype=np.float32)
    assert np.array_equal(called_with_array, expected_array)
