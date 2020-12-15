# Cloud Run ML API Demo

![CI](https://github.com/nm523/cloudrun-ml-demo/workflows/CI/badge.svg)

This repo contains a minimal example of using Github to deploy a machine learning model to Google Cloud Run. It uses [ONNX](https://onnx.ai/) to save and load the model and [falcon](https://falconframework.org/) to serve it.

## Example

```python
import pprint
import requests

payload = {
    "sepal_length_cm": 1,
    "sepal_width_cm": 2,
    "petal_length_cm": 3,
    "petal_width_cm": 4
}

resp = requests.post("https://my-cloud-run-url/predict", json=payload)
pprint.pprint(resp.json())
```

Outputs:
```python
{'predictions': [{'class_name': 'Iris Setosa',
                  'class_no': 0,
                  'score': -0.07042217254638672},
                 {'class_name': 'Iris Versicolour',
                  'class_no': 1,
                  'score': -2.1902551651000977},
                 {'class_name': 'Iris Virginica',
                  'class_no': 2,
                  'score': 2.2606773376464844}]}
```

## Tests

* [Tox](https://tox.readthedocs.io/en/latest/) is used to orchestrate the tests.
* The `dev-requirements.txt` file contains useful libraries for managing the code.

## Improvements

There are several improvements that can be considered, for example:

* Logging, using libraries such as [structlog](https://www.structlog.org/en/stable/).
* Making sure that the raw scores from the ONNX model are actual probabilities.
* Better tests and documentation (forever).
