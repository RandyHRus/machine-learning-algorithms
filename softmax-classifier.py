class SoftmaxLoss(FunObj):
    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        n, d = X.shape # 500,3
        k = len(np.unique(y))
       
        """YOUR CODE HERE FOR Q3.4"""
        # Hint: you may want to use NumPy's reshape() or flatten()
        # to be consistent with our matrix notation.
        f = 0
        w = np.reshape(w, (k,d))
        for i in range(n):
            wt = w[int(y[i])].T
            f -= wt @ X[i]
            sumk = 0
            for c in range(k):
                sumk += np.exp(w[c].T @ X[i])
                
            f += np.log(sumk) 
                
        
        
        g = np.zeros((k,d))
    
        for c in range(k):
            for j in range(d):
                sum = 0
                for i in range(n):
                    p=0
                    wx = w[c].T @ X[i]
                    sumc = 0
                    for cp in range(k):
                        sumc += np.exp(w[cp].T @ X[i])    

                    p = np.exp(wx)/ sumc
                    if(y[i] == c):
                        p -= 1

                    p *= X[i,j]
                    sum += p
                
                g[c,j] = sum

        g = g.flatten()
        return f, g

class MulticlassLogRegClassifier(LogRegClassifier):
    """
    LogRegClassifier's extention for multiclass classification.
    The constructor method and optimize() are inherited, so
    all you need to implement are fit() and predict() methods.
    """

    def fit(self, X, y):
        n, d = X.shape
        y_classes = np.unique(y)
        k = len(y_classes)
        assert set(y_classes) == set(range(k))  # check labels are {0, 1, ..., k-1}
        self.y_classes = len(y_classes)

        # Initial guesses for weights
        W = np.zeros([k, d])
        wflat = W.flatten()
        # Correctness check
        self.loss_fn.check_correctness(wflat, X, y.astype(np.float32))

        ytmp = y.copy().astype(float)
        W, self.fs, self.gs, self.ws = self.optimize(wflat, X, ytmp)

        self.W = W


    def predict(self, X_hat):
        n, d = X_hat.shape
        
        wreshape = np.reshape(self.W, (self.y_classes, d)).T 
        return np.argmax(X_hat @ wreshape, axis=1)