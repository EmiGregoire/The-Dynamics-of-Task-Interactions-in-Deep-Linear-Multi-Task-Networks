
import numpy as np

class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass

# This class QQT_lambda_balanced models the solution to the Fukumizu's QQt expression in the lambda balanced case. 
# It initializes with weights and data, calculates singular value
# decomposition, and performs forward propagation for
# one timestep, updating the QQT matrix accordingly. Handling unequal input output
#is a work in progress
class QQT_lambda_balanced:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):
        # Calculate lambda from initial weights
        self.lmda = (init_w2.T @ init_w2 - init_w1 @ init_w1.T)[0][0] 

        self.weights_only = weights_only
        self.batch_size = X.shape[0]

        # Set input, output, and hidden dimensions
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.hidden_dim = init_w2.shape[1]

        # Create identity matrix based on input or output dimensions
        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 
        
        # Calculate sigma_yx_tilde
        sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

        # Perform Singular Value Decomposition (SVD) on sigma_yx_tilde
        U_, S_, Vt_ = np.linalg.svd(sigma_yx_tilde)
        V_ = Vt_.T 

        # Construct matrix F based on lambda and sigma_yx_tilde
        self.F = np.vstack([
            np.hstack([-self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
            np.hstack([sigma_yx_tilde, self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
        ]) 

        self.U_, self.S_, self.V_ = U_, np.diag(S_), V_
        
        # Calculate dimension difference between input and output
        self.dim_diff = np.abs(self.input_dim - self.output_dim)

        # Handle cases where input and output dimensions are different
        if self.input_dim < self.output_dim:
            U_hat = U_[:, self.input_dim:]
            V_hat = np.zeros((self.input_dim, self.dim_diff))
            U_ = U_[:, :self.input_dim]

        elif self.input_dim > self.output_dim:
            U_hat = np.zeros((self.output_dim, self.dim_diff))
            V_hat = V_[:, self.output_dim:]
            V_ = V_[:, :self.output_dim]
        
        else:
            U_hat = None
            V_hat = None

        self.U_hat, self.V_hat = U_hat, V_hat
        self.U_, self.V_ = U_, V_

        # Perform SVD on the product of initial weights
        U, S, Vt = np.linalg.svd(init_w2 @ init_w1)
        self.U, self.S, self.V = U, S, Vt.T

        # Calculate inverse of singular values
        self.S_inv = np.diag(1. / np.diag(self.S_))

        self.S_ = np.diag(self.S_)

        # Calculate intermediate matrices X and A based on lambda and singular values
        self.X = (np.sqrt(self.lmda**2 + 4 * self.S_**2) - 2 * self.S_) / self.lmda
        self.A = 1 / (np.sqrt(1 + self.X**2))

        self.X = np.diag(self.X)
        self.A = np.diag(self.A)

        # Calculate secondary singular values
        self.S2 = np.sqrt((self.lmda + np.sqrt(self.lmda**2 + 4 * self.S**2)) / 2)

        _, s1, _ = np.linalg.svd(init_w1)
        _, s2, _ = np.linalg.svd(init_w2)

        # Initialize S1 and S2 matrices based on singular values
        self.S1 = np.zeros((self.hidden_dim, self.input_dim))
        self.S1[:len(s1), :len(s1)] = np.diag(s1)

        self.S2 = np.zeros((self.output_dim, self.hidden_dim))
        self.S2[:len(s2), :len(s2)] = np.diag(s2)

        # Calculate matrices B and C based on SVD components and intermediate matrices
        self.B = self.S2.T @ U.T @ U_ @ (self.X @ self.A + self.A) + self.S1 @ Vt @ V_ @ (self.A - self.X @ self.A)
        self.C = self.S2.T @ U.T @ U_ @ (self.A - self.X @ self.A) - self.S1 @ Vt @ V_ @ (self.X @ self.A + self.A)

        self.sign = np.sign(self.output_dim - self.input_dim)

        # Calculate eigenvalues and their inverses
        self.eval = np.sqrt(self.S_**2 + self.lmda**2 / 4)
        self.eval_inv = np.diag(1. / self.eval)
        self.eval_extra_dim = self.sign * self.lmda / 2 * np.ones(self.dim_diff)
        self.eval_extra_dim_inv = np.diag(1 / self.eval_extra_dim)

        # Check if matrix B is singular and raise an exception if so
        if np.isclose(np.linalg.det(self.B), 0):
            print('init_w1: ', init_w1)
            print('init_w2: ', init_w2)
            print('sigma_yx: ', sigma_yx_tilde)
            print('B: ', self.B)
            raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")

        self.B_inv = np.linalg.inv(self.B)
        
        self.t = 0

    def forward(self, learning_rate):
        # Performs forward propagation for one timestep

        time_step = self.t * learning_rate

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

        # Calculate exponential matrices for eigenvalues
        e_eval_st_inv = np.diag(np.exp(-1. * self.eval * time_step))
        e_eval_2st_inv = np.diag(np.exp(-2. * self.eval * time_step))
        e_eval_st_extra_dim = np.diag(np.exp(1. * self.eval_extra_dim * time_step))
        e_eval_2st_extra_dim = np.diag(np.exp(2. * self.eval_extra_dim * time_step))

        if self.U_hat is None and self.V_hat is None:
            Z = np.vstack([
                self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
                self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv)
            ])
            center_add = 0.
        else:
            Z_add = np.vstack([
                self.V_hat @ e_eval_st_extra_dim @ self.V_hat.T @ self.V @ self.S1.T @ self.B_inv.T @ e_eval_st_inv,
                self.U_hat @ e_eval_st_extra_dim @ self.U_hat.T @ self.U @ self.S2 @ self.B_inv.T @ e_eval_st_inv
            ])

            Z = np.vstack([
                self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
                self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv)
            ])

            Z = Z + Z_add

            center_add = (2 * e_eval_st_extra_dim @ self.B_inv
                          @ (self.S1 @ self.V.T @ self.V_hat @ (e_eval_2st_extra_dim - i) @ self.eval_extra_dim_inv @ self.V_hat.T @ self.V @ self.S1.T
                          + self.S2.T @ self.U.T @ self.U_hat @ (e_eval_2st_extra_dim - i) @ self.eval_extra_dim_inv @ self.U_hat.T @ self.U @ self.S2)
            @ self.B_inv.T @ e_eval_2st_inv) 

        center_left = 4 * e_eval_st_inv @ self.B_inv @ self.B_inv.T @ e_eval_st_inv

        center_center = (i - e_eval_2st_inv) @ self.eval_inv

        center_right = e_eval_st_inv @ self.B_inv @ self.C @ (e_eval_2st_inv - i) @ self.eval_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv

        center = center_left + center_center - center_right + center_add

        # Perform Cholesky decomposition for stability
        L = np.linalg.cholesky(center)
        y = np.linalg.solve(L, Z.T)
        x = np.linalg.solve(L.T, y)
        qqt = x.T @ Z.T
        
        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t += 1
        return qqt 
