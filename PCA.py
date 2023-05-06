# principle component analysis

def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    utils.savefig("animals_matrix.png", fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(trait_names[i], xy=xy)
    utils.savefig("animals_random.png", fig)

    """YOUR CODE HERE FOR Q3"""
    #raise NotImplementedError()
    encoder = PCAEncoder(2)
    encoder.fit(X_train)
    Z = encoder.encode(X_train)
    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])
    for i in range(n):
        xy = Z[i, [0, 1]]
        ax.annotate(animal_names[i], xy=xy)
    utils.savefig("animals_random_encoded.png", fig)