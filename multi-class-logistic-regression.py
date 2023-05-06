class LogRegClassifierOneVsAll(LogRegClassifier):
    """
    Uses a function object and an optimizer.
    """

    def fit(self, X, y):
        n, d = X.shape
        y_classes = np.unique(y)
        k = len(y_classes)
        assert set(y_classes) == set(range(k))  # check labels are {0, 1, ..., k-1}

        # quick check that loss_fn is implemented correctly
        self.loss_fn.check_correctness(np.zeros(d), X, (y == 1).astype(np.float32))

        # Initial guesses for weights
        W = np.zeros([k, d])

        """YOUR CODE HERE FOR Q3.2"""
        # NOTE: make sure that you use {-1, 1} labels y for logistic regression,
        #       not {0, 1} or anything else.
        #raise NotImplementedError()
        
        for i in range(y_classes.size):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1
            W[i], self.fs, self.gs, self.ws = self.optimize(W[i], X, ytmp)

        self.W = W


    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)