def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    cv_accs = np.zeros(len(ks))

    idx=0
    n, d = X.shape
    for k in ks:
        errors = np.zeros(10)
        for j in range(10):
            i = j + 1
            model = KNN(k)
            masks = np.ones(n, dtype=bool)

            masks[:int(i * np.floor(n/10))] = 0
            masks[:int((i-1) * np.floor(n/10))] = 1

            X_masked = X[~masks,:]
            y_masked = y[:, np.newaxis]
            y_masked = y_masked[~masks,:]

            X_valid = X[masks,:]
            y_valid = y[:, np.newaxis]
            y_valid = y_valid[masks,:]

            model.fit(X_masked, y_masked)
            y_pred = model.predict(X_valid)
            
            error = np.mean(y_pred != y_valid)
            errors[j] = error

        error_mean = np.mean(errors)
        cv_accs[idx] = error_mean
        idx+=1