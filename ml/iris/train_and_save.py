"""Train and serialise an Iris model.

Example taken from: http://onnx.ai/sklearn-onnx/.
"""

import numpy
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Train a model.
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = LogisticRegression()
clr.fit(X_train, y_train)

# Convert into ONNX format
initial_type = [("float_input", FloatTensorType([None, 4]))]
options = {
    id(clr): {"raw_scores": True}
}  # Slightly different from the tutorial, allow the API to export raw scores.
onx = convert_sklearn(clr, initial_types=initial_type, options=options)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with ONNX Runtime
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
