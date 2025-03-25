import numpy as np

class LassoHomotopyModel:
    def __init__(self, lambda_reg=0.1):
        self.lambda_reg = lambda_reg
        self.theta = None
        self.active_set = None
        self.signs = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def preprocess(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std == 0] = 1.0
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std == 0:
            self.y_std = 1.0
        X_scaled = (X - self.X_mean) / self.X_std
        y_scaled = (y - self.y_mean) / self.y_std
        return X_scaled, y_scaled

    def inverse_preprocess_theta(self):
        if self.X_std is None or self.y_std is None:
            return self.theta
        theta_adjusted = self.theta * self.y_std / self.X_std
        return theta_adjusted

    def fit(self, X, y):
        X_scaled, y_scaled = self.preprocess(X, y)
        n_samples, n_features = X_scaled.shape
        self.theta = np.zeros(n_features)
        self.active_set = []
        self.signs = []

        lambda_max = np.max(np.abs(X_scaled.T @ y_scaled))
        lambda_current = lambda_max
        print(f"Initial lambda_max: {lambda_max}, Target lambda_reg: {self.lambda_reg}")

        max_iterations = 100
        iteration = 0
        while lambda_current > self.lambda_reg and iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}: lambda_current = {lambda_current}")
            residuals = y_scaled - X_scaled @ self.theta
            gradient = X_scaled.T @ residuals
            print(f"Gradient norm: {np.linalg.norm(gradient)}")
            print(f"Gradient: {gradient}")
            print(f"Active set: {self.active_set}")
            print(f"Current theta: {self.theta}")

            if not self.active_set:
                idx = np.argmax(np.abs(gradient))
                self.active_set.append(idx)
                self.signs.append(np.sign(gradient[idx]))
                print(f"Added feature {idx} to active set with sign {self.signs[-1]}")
                lambda_current = np.abs(gradient[idx])
                continue

            X_active = X_scaled[:, self.active_set]
            signs_active = np.array([np.sign(gradient[idx]) for idx in self.active_set])
            G = X_active.T @ X_active + 1e-6 * np.eye(len(self.active_set))
            try:
                direction_active = np.linalg.solve(G, X_active.T @ y_scaled - lambda_current * signs_active)
            except np.linalg.LinAlgError:
                print("Matrix singular, breaking")
                break

            direction_theta = np.zeros(n_features)
            for i, idx in enumerate(self.active_set):
                direction_theta[idx] = direction_active[i]
            print(f"Direction theta: {direction_theta}")

            # Gamma for removal
            gamma_remove = np.inf
            remove_idx = None
            for i, idx in enumerate(self.active_set):
                if direction_theta[idx] != 0:
                    gamma = -self.theta[idx] / direction_theta[idx]
                    if gamma > 0 and gamma < gamma_remove:
                        gamma_remove = gamma
                        remove_idx = idx
            print(f"Gamma remove: {gamma_remove}, Remove idx: {remove_idx}")

            # Gamma for addition
            gamma_add = np.inf
            add_idx = None
            add_sign = 0
            non_active = [i for i in range(n_features) if i not in self.active_set]
            correlation = X_scaled.T @ (X_scaled @ direction_theta)
            for idx in non_active:
                grad_idx = gradient[idx]
                delta = correlation[idx]
                if delta != 0:
                    if grad_idx > 0:
                        gamma = (lambda_current - grad_idx) / delta
                    else:
                        gamma = (lambda_current + grad_idx) / delta
                    if gamma > 0 and gamma < gamma_add:
                        gamma_add = gamma
                        add_idx = idx
                        add_sign = np.sign(grad_idx)
            print(f"Gamma add: {gamma_add}, Add idx: {add_idx}, Add sign: {add_sign}")

            gamma = min(gamma_remove, gamma_add)
            if gamma == np.inf or gamma <= 0:
                print("No valid transition, adjusting lambda")
                gamma = 0.1
                lambda_current = max(self.lambda_reg, lambda_current * 0.9)
            else:
                print(f"Gamma: {gamma}")
                gamma = min(gamma, 1.0)  # Cap gamma to control steps
                lambda_decrement = gamma * (lambda_current - self.lambda_reg)
                lambda_current = max(self.lambda_reg, lambda_current - lambda_decrement)

            self.theta += gamma * direction_theta
            print(f"Updated theta: {self.theta}")
            print(f"Updated lambda_current: {lambda_current}")

            if gamma_remove < gamma_add and remove_idx is not None:
                print(f"Removing feature {remove_idx}")
                i = self.active_set.index(remove_idx)
                self.active_set.pop(i)
                self.signs.pop(i)
                self.theta[remove_idx] = 0
            elif gamma_add <= gamma_remove and add_idx is not None:
                print(f"Adding feature {add_idx} with sign {add_sign}")
                self.active_set.append(add_idx)
                self.signs.append(add_sign)

        self.theta = self.inverse_preprocess_theta()
        print(f"Final theta: {self.theta}")
        return LassoHomotopyResults(self.theta)

class LassoHomotopyResults:
    def __init__(self, theta):
        self.theta = theta

    def predict(self, X):
        return X @ self.theta