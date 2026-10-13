import numpy as np


def _normalize_task_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)
    return weights / np.sum(weights)


def _pcgrad_two_task(g1, g2, eps=1e-12):
    g1_orig = np.asarray(g1, dtype=np.float64).copy()
    g2_orig = np.asarray(g2, dtype=np.float64).copy()
    v1 = g1_orig.reshape(-1)
    v2 = g2_orig.reshape(-1)
    dot12 = float(v1 @ v2)
    if dot12 >= 0.0:
        return g1_orig, g2_orig
    g1_proj = g1_orig - (dot12 / (np.linalg.norm(v2) ** 2 + eps)) * g2_orig
    g2_proj = g2_orig - (dot12 / (np.linalg.norm(v1) ** 2 + eps)) * g1_orig
    return g1_proj, g2_proj


def _compute_next_task_weights(
    weighting_mode,
    current_weights,
    gradmag1,
    gradmag2,
    cos_sim,
    loss1,
    loss2,
    state,
    main_task=1,
    gradnorm_alpha=1.5,
    gradnorm_lr=0.025,
):
    eps = 1e-12
    mode = weighting_mode.lower()

    if mode == "none":
        return np.array([1.0, 1.0], dtype=np.float64), state

    if mode == "cossim":
        aux_weight = max(0.0, float(cos_sim))
        if main_task == 1:
            return _normalize_task_weights(np.array([1.0, aux_weight], dtype=np.float64)), state
        if main_task == 2:
            return _normalize_task_weights(np.array([aux_weight, 1.0], dtype=np.float64)), state
        raise ValueError("main_task must be 1 or 2 for cossim weighting.")

    if mode == "gradnorm":
        losses = np.array([max(loss1, eps), max(loss2, eps)], dtype=np.float64)
        if state.get("initial_losses") is None:
            state["initial_losses"] = losses.copy()
            current_weights = np.array([0.5, 0.5], dtype=np.float64)

        base_grad_norms = np.array([gradmag1, gradmag2], dtype=np.float64)
        grad_norms = current_weights * base_grad_norms
        grad_norm_mean = np.mean(grad_norms)

        loss_ratios = losses / state["initial_losses"]
        inverse_train_rates = loss_ratios / np.mean(loss_ratios)
        targets = grad_norm_mean * np.power(inverse_train_rates, gradnorm_alpha)

        grad_w = np.sign(grad_norms - targets) * base_grad_norms
        next_weights = current_weights - gradnorm_lr * grad_w
        return _normalize_task_weights(next_weights), state

    raise ValueError("weighting_mode must be one of {'none', 'gradnorm', 'cossim'}.")


class LinearNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1=None, init_w2=None):

        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x):
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):

        forward = self.W2 @ self.W1 @ x
        dW1 = 1 / x.shape[1] * self.W2.T @ (forward - y) @ x.T
        dW2 = 1 / x.shape[1] * (forward - y) @ x.T @ self.W1.T

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1

    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []
        for _ in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
        return w1s, w2s, losses

    def train_fast(self, X_train, Y_train, epochs, learning_rate):
        losses = []
        for _ in range(epochs):
            loss = (
                1
                / (2 * X_train.shape[1])
                * np.linalg.norm(self.W2 @ self.W1 @ X_train - Y_train, ord="fro") ** 2
            )
            losses.append(loss)
            self.backward(X_train, Y_train, learning_rate)
        return losses


class MTL_linearnetwork:
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim_1,
        out_dim_2,
        init_ws=None,
        init_w1=None,
        init_w2=None,
    ):

        if init_ws is not None and init_w1 is not None and init_w2 is not None:
            self.WS = init_ws.copy()
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.WS = np.random.randn(hidden_dim, in_dim)
            self.W1 = np.random.randn(out_dim_1, hidden_dim)
            self.W2 = np.random.randn(out_dim_2, hidden_dim)

        self.taskweight1_history = []
        self.taskweight2_history = []
        self.task_weights = np.array([1.0, 1.0], dtype=np.float64)
        self.next_task_weights = np.array([1.0, 1.0], dtype=np.float64)
        self.postsurg_cossim_history = []
        self.postsurg_gradmag1_history = []
        self.postsurg_gradmag2_history = []

    def forward(self, x):
        self.z = self.WS @ x
        self.T1 = self.W1 @ self.z
        self.T2 = self.W2 @ self.z
        return self.T1, self.T2

    def backward(
        self,
        x,
        y1,
        y2,
        learning_rate,
        loss1,
        loss2,
        weight_state,
        weighting_mode="none",
        gradient_surgery="none",
        main_task=1,
        gradnorm_alpha=1.5,
        gradnorm_lr=0.025,
    ):

        forwardT1 = self.W1 @ self.WS @ x
        forwardT2 = self.W2 @ self.WS @ x

        m = x.shape[1]
        gWS1 = (1 / m) * (self.W1.T @ (forwardT1 - y1) @ x.T)
        gWS2 = (1 / m) * (self.W2.T @ (forwardT2 - y2) @ x.T)
        base_dW1 = (1 / m) * (forwardT1 - y1) @ x.T @ self.WS.T
        base_dW2 = (1 / m) * (forwardT2 - y2) @ x.T @ self.WS.T

        eps = 1e-12
        v1 = gWS1.reshape(-1)
        v2 = gWS2.reshape(-1)
        self.cos_sim = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
        self.gradmag1 = np.linalg.norm(v1)
        self.gradmag2 = np.linalg.norm(v2)

        current_weights = np.asarray(weight_state["task_weights"], dtype=np.float64).copy()
        self.task_weights = current_weights.copy()

        shared_g1 = current_weights[0] * gWS1.copy()
        shared_g2 = current_weights[1] * gWS2.copy()
        if gradient_surgery.lower() == "pcgrad":
            shared_g1, shared_g2 = _pcgrad_two_task(shared_g1, shared_g2)
        elif gradient_surgery.lower() != "none":
            raise ValueError("gradient_surgery must be one of {'none', 'pcgrad'}.")

        sv1 = shared_g1.reshape(-1)
        sv2 = shared_g2.reshape(-1)
        self.postsurg_gradmag1 = np.linalg.norm(sv1)
        self.postsurg_gradmag2 = np.linalg.norm(sv2)
        self.postsurg_cos_sim = (sv1 @ sv2) / (
            self.postsurg_gradmag1 * self.postsurg_gradmag2 + eps
        )

        dWS = shared_g1 + shared_g2
        dW1 = current_weights[0] * base_dW1
        dW2 = current_weights[1] * base_dW2

        next_weights, weight_state = _compute_next_task_weights(
            weighting_mode=weighting_mode,
            current_weights=current_weights,
            gradmag1=self.gradmag1,
            gradmag2=self.gradmag2,
            cos_sim=self.cos_sim,
            loss1=loss1,
            loss2=loss2,
            state=weight_state,
            main_task=main_task,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_lr=gradnorm_lr,
        )
        self.next_task_weights = next_weights.copy()
        weight_state["task_weights"] = next_weights.copy()

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1
        self.WS -= learning_rate * dWS
        return weight_state

    def trainMTL(
        self,
        X_train,
        Y1_train,
        Y2_train,
        epochs,
        learning_rate,
        trackgradients=False,
        weighting_mode="none",
        gradient_surgery="none",
        main_task=1,
        gradnorm_alpha=1.5,
        gradnorm_lr=0.025,
        progress_callback=None,
        save_weight_history=False,
    ):
        wss = []
        w1s = []
        w2s = []
        losses1 = []
        losses2 = []

        cossimlist = []
        gradmag1list = []
        gradmag2list = []
        self.taskweight1_history = []
        self.taskweight2_history = []
        self.postsurg_cossim_history = []
        self.postsurg_gradmag1_history = []
        self.postsurg_gradmag2_history = []
        init_weights = np.array([1.0, 1.0], dtype=np.float64)
        weight_state = {"initial_losses": None, "task_weights": init_weights}

        for step in range(epochs):
            forwardT1, forwardT2 = self.forward(X_train)
            loss1 = np.mean((forwardT1 - Y1_train) ** 2)
            losses1.append(loss1)
            loss2 = np.mean((forwardT2 - Y2_train) ** 2)
            losses2.append(loss2)

            if save_weight_history:
                wss.append(self.WS.copy())
                w1s.append(self.W1.copy())
                w2s.append(self.W2.copy())

            weight_state = self.backward(
                X_train,
                Y1_train,
                Y2_train,
                learning_rate,
                loss1=loss1,
                loss2=loss2,
                weight_state=weight_state,
                weighting_mode=weighting_mode,
                gradient_surgery=gradient_surgery,
                main_task=main_task,
                gradnorm_alpha=gradnorm_alpha,
                gradnorm_lr=gradnorm_lr,
            )

            self.taskweight1_history.append(float(self.task_weights[0]))
            self.taskweight2_history.append(float(self.task_weights[1]))
            self.postsurg_cossim_history.append(float(self.postsurg_cos_sim))
            self.postsurg_gradmag1_history.append(float(self.postsurg_gradmag1))
            self.postsurg_gradmag2_history.append(float(self.postsurg_gradmag2))

            if trackgradients:
                cossimlist.append(self.cos_sim)
                gradmag1list.append(self.gradmag1)
                gradmag2list.append(self.gradmag2)

            if progress_callback is not None:
                progress_callback(
                    {
                        "step": step + 1,
                        "total_steps": epochs,
                        "loss1": float(loss1),
                        "loss2": float(loss2),
                        "taskweight1": float(self.task_weights[0]),
                        "taskweight2": float(self.task_weights[1]),
                        "cossim": float(self.cos_sim),
                        "gradmag1": float(self.gradmag1),
                        "gradmag2": float(self.gradmag2),
                    }
                )

        if trackgradients:
            return wss, w1s, w2s, losses1, losses2, cossimlist, gradmag1list, gradmag2list

        return wss, w1s, w2s, losses1, losses2
