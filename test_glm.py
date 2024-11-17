import numpy as np
from GLM_models import NormalGLM, BernoulliGLM, PoissonGLM

def test_glms():
    # Generate dummy data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # Two predictors
    y_normal = X @ np.array([1.5, -2.0]) + 0.5 + np.random.normal(0, 1, 100)  # Normal
    y_bernoulli = (np.random.rand(100) > 0.5).astype(int)  # Bernoulli
    y_poisson = np.random.poisson(np.exp(X @ np.array([0.3, 0.7])))  # Poisson

    # Test NormalGLM
    print("\nTesting NormalGLM")
    normal_model = NormalGLM(X, y_normal)
    normal_model.fit()
    print("Predictions:", normal_model.predict(X[:5]))

    # Test BernoulliGLM
    print("\nTesting BernoulliGLM")
    bernoulli_model = BernoulliGLM(X, y_bernoulli)
    bernoulli_model.fit()
    print("Predictions:", bernoulli_model.predict(X[:5]))

    # Test PoissonGLM
    print("\nTesting PoissonGLM")
    poisson_model = PoissonGLM(X, y_poisson)
    poisson_model.fit()
    print("Predictions:", poisson_model.predict(X[:5]))

if __name__ == "__main__":
    test_glms()
