import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from scipy.stats import norm, bernoulli, poisson

# Base Class: Generalized Linear Model
class GeneralizedLinearModel(ABC):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.params = None

    @abstractmethod
    def link_function(self, mu):
        pass

    @abstractmethod
    def negative_log_likelihood(self, params):
        pass

    def fit(self):
        init_params = np.zeros(self.X.shape[1])
        result = minimize(self.negative_log_likelihood, init_params, method='BFGS')
        self.params = result.x
        print("Optimization finished. Parameters: ", self.params)

    def predict(self, X_new):
        eta = np.dot(X_new, self.params)
        return self.link_function(eta)

# Subclass: Normal Distribution
class NormalGLM(GeneralizedLinearModel):
    def link_function(self, mu):
        return mu

    def negative_log_likelihood(self, params):
        eta = np.dot(self.X, params)
        mu = self.link_function(eta)
        nll = -np.sum(norm.logpdf(self.y, loc=mu, scale=1))
        return nll

# Subclass: Bernoulli Distribution
class BernoulliGLM(GeneralizedLinearModel):
    def link_function(self, mu):
        return 1 / (1 + np.exp(-mu))

    def negative_log_likelihood(self, params):
        eta = np.dot(self.X, params)
        mu = self.link_function(eta)
        nll = -np.sum(self.y * np.log(mu) + (1 - self.y) * np.log(1 - mu))
        return nll

# Subclass: Poisson Distribution
class PoissonGLM(GeneralizedLinearModel):
    def link_function(self, mu):
        return np.exp(mu)

    def negative_log_likelihood(self, params):
        eta = np.dot(self.X, params)
        mu = self.link_function(eta)
        nll = -np.sum(poisson.logpmf(self.y, mu))
        return nll