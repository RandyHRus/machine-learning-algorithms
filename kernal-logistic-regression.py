def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    for i in range(len(sigmas)):
        for j in range(len(lammys)):
            loss_fn = KernelLogisticRegressionLossL2(lammy=lammys[j])
            optimizer = GradientDescentLineSearch()
            kernel = GaussianRBFKernel(sigmas[i])
            klr_model = KernelClassifier(loss_fn, optimizer, kernel)
            klr_model.fit(X_train, y_train)
            train_errs[i,j] = np.mean(klr_model.predict(X_train) != y_train)
            val_errs[i,j] = np.mean(klr_model.predict(X_val) != y_val)

    #Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    utils.savefig("logRegRBF_grids.png", fig)