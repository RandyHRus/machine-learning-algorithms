def q2_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    v = np.append(np.ones(400), 0.1*np.ones(100))
    model = linear_models.WeightedLeastSquares()
    model.fit(X, y, v)
   

class WeightedLeastSquares(LeastSquares):
    def fit(self, X, y, v):
            V = np.diag(v)
            self.w = solve( X.T @ V @ X, X.T @ V @ y)