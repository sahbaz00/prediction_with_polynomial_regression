import numpy as np

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self.coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return X_poly @ self.coefficients

    def _polynomial_features(self, X):
        m, n = X.shape
        X_poly = np.ones((m, 1))

        for d in range(1, self.degree + 1):
            for i in range(n):
                X_poly = np.column_stack((X_poly, np.power(X[:, i], d)))

        return X_poly[:, 1:]