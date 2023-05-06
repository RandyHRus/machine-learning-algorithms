def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.1"""
    batch_size = [1,10,100]
    stepSize = 0.0003

    for i in range(len(batch_size)):
        loss_fn = LeastSquaresLoss()
        optimizer = StochasticGradient(base_optimizer = GradientDescent(), learning_rate_getter=ConstantLR(stepSize),batch_size= batch_size[i], max_evals=10 )
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train, y_train)
        print("batch size= ", batch_size[i])
        print(model.fs)  # ~700 seems to be the global minimum.

        print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
        print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    utils.savefig("gd_line_search_curve.png", fig)


class ConstantLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier


class InverseLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier/self.num_evals


class InverseSquaredLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier/(self.num_evals**2)


class InverseSqrtLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier/np.sqrt(self.num_evals)