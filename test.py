from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def test_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    assert model.score(X, y) > 0.9
