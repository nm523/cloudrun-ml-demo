# Cloud Run ML API Demo

![CI](https://github.com/nm523/cloudrun-ml-demo/workflows/CI/badge.svg)

This repo contains a minimal example of using Github to deploy a machine learning model to Google Cloud Run.

## Example

```python
import requests

payload = {
    "sepal_length_cm": 1,
    "sepal_width_cm": 2,
    "petal_length_cm": 3,
    "petal_width_cm": 4
}

resp = requests.post("https://my-cloud-run-url/predict", json=payload)
