# K-nearest-neighbours
def predict(self, X_hat):
    squared = euclidean_dist_squared(X_hat,self.X)
    n, d = squared.shape
    y_pred = np.zeros(n)
    for i in range(n):
        karray = np.argsort(squared[i])
        k_sorted = karray[:self.k]
        k_result = np.zeros(self.k)
        for j in range(self.k):
            k_result[j] = self.y[k_sorted[j]]
            
        knn = utils.mode(k_result)
        
        y_pred[i] = knn

    return y_pred