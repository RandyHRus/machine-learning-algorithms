def error(self, X, y, means):
    sum = 0

    n, d = X.shape
    for i in range(n):
        for j in range(d):
            sum += (X[i,j] - means[y[i], j])**2
    
    return sum