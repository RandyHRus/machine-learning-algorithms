def predict(self, X):      
    n, d = np.shape(X)

    predictions = np.zeros(shape=(n, self.num_trees))

    for i in range(self.num_trees):
        r = self.list_of_trees[i].predict(X)
        predictions[:, i] = r

    modes = np.zeros(n)
    for i in range(n):
        modes[i] = utils.mode(predictions[i])

    print(modes)
    return modes