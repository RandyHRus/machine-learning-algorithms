@handle("2.4.1")
def q2_4_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = FunObjRobustRegression()
    optimizer = OptimizerGradientDescentLineSearch(max_evals=100, verbose=False)
    model = linear_models.LinearModelGradientDescent(fun_obj, optimizer, check_correctness_yes=True)
    model.fit(X, y)

    utils.test_and_plot(
        model,
        X,
        y,
        title="Robust Linear Regression with Gradient Descent",
        filename="robust_least_squares_gd.pdf",
    )
    
class FunObjRobustRegression(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        f = np.sum(np.log(np.exp((X @ w) - y) + np.exp(y -(X @ w))))
       
        n,d = X.shape
        s = 0

        for i in range(n):
            numerator = X[i] * (np.exp(w * X[i] - y[i]) - np.exp(y[i] - w * X[i]))
            denominator = np.exp(w * X[i] - y[i]) + np.exp(y[i]-w*X[i])
            s += (numerator/denominator)

        g = s

        return f, g