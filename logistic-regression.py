class LogisticRegressionLossL2(LogisticRegressionLoss):
    def __init__(self, lammy):
        super().__init__()
        self.lammy = lammy

    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y_i are in {-1, 1}

        # Calculate the function value
        f = np.sum(np.log(1 + np.exp(-yXw))) + (self.lammy/2) * np.linalg.norm(w) ** 2

        # Calculate the gradient value
        s = -y / (1 + np.exp(yXw)) 
        g = X.T @ s + (self.lammy * w)

        return f, g
    

def q2_2():
    data = utils.load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    lammies = [0.01, 0.1,1,10]
    for l in lammies:
        fun_obj = LogisticRegressionLossL1(l)
        optimizer = GradientDescentLineSearchProxL1(max_evals=400, verbose=False, lammy=l)
        model = linear_models.LogRegClassifier(fun_obj, optimizer)
        model.fit(X, y)

        print("lammy=",l)
        train_err = utils.classification_error(model.predict(X), y)
        print(f"LogReg Training error: {train_err:.3f}")

        val_err = utils.classification_error(model.predict(X_valid), y_valid)
        print(f"LogReg Validation error: {val_err:.3f}")

        print(f"# nonZeros: {np.sum(model.w != 0)}")
        print(f"# function evals: {optimizer.num_evals}")

class LogisticRegressionLossL1(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1 + np.exp(-yXw))) + (self.lammy * np.linalg.norm(w,ord = 1))

        # Calculate the gradient value
        s = -y / (1 + np.exp(yXw)) 
        g = X.T @ s + self.lammy
        return f, g