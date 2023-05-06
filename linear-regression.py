class LeastSquaresBias:
    "Least Squares with a bias added"
    def fit(self, X, y):
        n, d = X.shape
        Z = np.hstack((np.ones(n).reshape((n, 1)),X))

        self.v = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X_pred):
        n, d = X_pred.shape
        Z_pred = np.hstack((np.ones(n).reshape((n, 1)),X_pred))

        return Z_pred @ self.v
    

class LeastSquaresPoly:
    "Least Squares with polynomial basis"
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self, X, y):
        Z = self._poly_basis(X)
        self.v = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X_pred):
        Z = self._poly_basis(X_pred)
        return Z @ self.v

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def _poly_basis(self, X):
        n, d = X.shape
        Z = np.ones(n).reshape((n, 1))
        if self.p == 0 :
            Z = X
        else:
            for i in range(self.p):
                j = i + 1
                Xj = X**j
                Z = np.hstack((Z, Xj))

        return Z