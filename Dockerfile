FROM python:3.8-slim-buster

WORKDIR /app

# Install the requirements.
COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]

# Copy over application code.
COPY app app

# Copy the model over and set the environment variable so the app can find it.
COPY ml/iris/model.onnx model.onnx
ENV IRIS_ONNX_MODEL_PATH=/app/model.onnx

# Copy over the gunicorn configuration last.
COPY gunicorn.conf.py gunicorn.conf.py

EXPOSE 8000
ENTRYPOINT [ "gunicorn", "app.main:api" ]
