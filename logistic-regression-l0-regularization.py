def fit(self, X, y):
        n, d = X.shape

        # Maintain the set of selected indices, as a boolean mask array.
        # We assume that feature 0 is a bias feature, and include it by default.
        selected = np.zeros(d, dtype=bool)
        selected[0] = True
        min_loss = np.inf
        self.total_evals = 0

        # We will hill-climb until a local discrete minimum is found.
        while not np.all(selected):
            old_loss = min_loss
            print(f"Epoch {selected.sum():>3}:", end=" ")

            best_feature = 0

            for j in range(d):
                if selected[j]:
                    continue

                selected_with_j = selected.copy()
                selected_with_j[j] = True

                score, gradient = self.loss_fn.evaluate(selected_with_j, X, y)
                if score < min_loss:
                    best_feature = j
                    min_loss = score
                self.total_evals += 1


            if min_loss < old_loss:  # something in the loop helped our model
                selected[best_feature] = True
                print(f"adding feature {best_feature:>3} - loss {min_loss:>7.3f}")
            else:
                print("nothing helped to add; done.")
                break
        else:  # triggers if we didn't break out of the loop
            print("wow, we selected everything")

        w_init = np.zeros(selected.sum())
        w_on_sub, *_ = self.optimize(w_init, X[:, selected], y)
        self.total_evals += self.optimizer.num_evals

        self.w = np.zeros(d)
        self.w[selected] = w_on_sub