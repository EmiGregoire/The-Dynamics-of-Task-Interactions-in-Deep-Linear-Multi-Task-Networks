import matplotlib.pyplot as plt
import numpy as np

class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass

class QQT_lambda_balanced_unequal:
    def __init__(self, init_w1, init_w2, X, Y, batch_size, weights_only=False,):

        self.lmda = (init_w2.T @ init_w2 - init_w1 @ init_w1.T)[0][0]
        self.hidden_dim = init_w1.shape[0]
        self.weights_only = weights_only

        self.batch_size = batch_size

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim)

        sigma_yx_tilde = 1 / self.batch_size * Y.T @ X

        U_, S_, Vt_ = np.linalg.svd(sigma_yx_tilde)
        V_ = Vt_.T

        self.F = np.vstack([
            np.hstack([- self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
            np.hstack([sigma_yx_tilde, self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
        ])

        self.U_, self.S_, self.V_ = U_, np.diag(S_), V_

        self.dim_diff = np.abs(self.input_dim - self.output_dim)

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
        # U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
        U, S, Vt = np.linalg.svd(init_w2 @ init_w1)
        self.U, self.S, self.V = U, S, Vt.T

        self.S_inv = np.diag(1. / np.diag(self.S_))

        self.S_ = np.diag(self.S_)

        # self.X = (np.sqrt(self.lmda ** 2 + 4 * self.S_ ** 2) - 2 * self.S_) / self.lmda
        # self.A = 1 / (np.sqrt(1 + self.X ** 2))

        # Calculate the numerator
        numerator = np.sqrt( np.sqrt(self.S_ ** 2+ (self.lmda ** 2 / 4)  )- self.S_)

        # Calculate the denominator
        denominator = np.sqrt(np.sqrt(self.S_ ** 2+ (self.lmda ** 2 / 4)  ) + self.S_)

        # Calculate the final result
        self.X =  numerator / denominator
        self.X = np.sign(self.lmda) * numerator / denominator
        #        print(tilde_X-self.X)
        #self.X_2 = np.diag( tilde_X )
        self.A = 1 / (np.sqrt(1 + self.X ** 2))

        self.X = np.diag(self.X)
        self.A = np.diag(self.A)


        _, s1, _ = np.linalg.svd(init_w1)
        _, s2, _ = np.linalg.svd(init_w2)

        self.S1 = np.zeros((self.hidden_dim, self.input_dim))
        self.S1_prime = np.zeros((self.hidden_dim, self.hidden_dim))
        self.S2 = np.zeros((self.output_dim, self.hidden_dim))
        self.S2_prime = np.zeros((self.hidden_dim, self.hidden_dim))

        self.dim_diff_hidden_output = self.hidden_dim - self.output_dim
        if self.dim_diff_hidden_output > 0:
            self.S1[:(len(s1) - self.dim_diff_hidden_output), :(len(s1) - self.dim_diff_hidden_output)] = np.diag(s1)[:(
                        len(s1) - self.dim_diff_hidden_output), :(len(s1) - self.dim_diff_hidden_output)]
            self.S1[(len(s1) - self.dim_diff_hidden_output):len(s1),
            (len(s1) - self.dim_diff_hidden_output):len(s1)] = np.diag(np.ones((self.dim_diff_hidden_output, 1)) * 0)
            self.S1_prime[:len(s1), :len(s1)] = np.diag(s1)
        else:
            self.S1[:len(s1), :len(s1)] = np.diag(s1)

        self.dim_diff_hidden_input = self.hidden_dim - self.input_dim
        if self.dim_diff_hidden_input > 0:
            self.S2[:(len(s2) - self.dim_diff_hidden_input), :(len(s2) - self.dim_diff_hidden_input)] = np.diag(s2)[:(
                        len(s2) - self.dim_diff_hidden_input), :(len(s2) - self.dim_diff_hidden_input)]
            self.S2[(len(s2) - self.dim_diff_hidden_input):len(s2),
            (len(s2) - self.dim_diff_hidden_input):len(s2)] = np.diag(
                np.ones((self.dim_diff_hidden_input, 1)) * 0)
            self.S2_prime[:len(s2), :len(s2)] = np.diag(s2)
        else:
            self.S2[:len(s2), :len(s2)] = np.diag(s2)

        #U, S, V = np.linalg.svd(init_w2 @ init_w1)
        #U_, S_, V_ = np.linalg.svd(sigma_yx_tilde)
        #self._B =  U.T @ U_ +  V.T @ V_
        # self._C = U.T @ U_ -  V.T @ V_
        #_B = U.T @ U_ + V.T @ V_
        #_C = U.T @ U_ - V.T @ V_

        #print('B_before')
        #print(np.diag( _B ))

        #print('C_before')
        #print(np.diag(_C))
        #        V1
        self.B = self.S2.T @ U.T @ U_ @ (self.X @ self.A + self.A) + self.S1 @ Vt @ V_ @ (self.A - self.X @ self.A)

        self.C = self.S2.T @ U.T @ U_ @ (self.A - self.X @ self.A) - self.S1 @ Vt @ V_ @ (self.X @ self.A + self.A)

        # self.B = self.S2.T @ U.T @ U_ @ (-self.X @ self.A + self.A) + self.S1 @ Vt @ V_ @ (self.A + self.X @ self.A)

        # self.C = self.S2.T @ U.T @ U_ @ (self.A + self.X @ self.A) - self.S1 @ Vt @ V_ @ (-self.X @ self.A + self.A)
        #        V2
        # self.B = self.S2.T @ U.T @ U_ @ (self.X @ self.A + self.A) + self.S1 @ Vt @ V_ @ (self.A - self.X @ self.A)

        # self.C = self.S2.T @ U.T @ U_ @ (self.A - self.X @ self.A) - self.S1 @ Vt @ V_ @ (self.X @ self.A + self.A)

        self.sign = np.sign(self.output_dim - self.input_dim)

        self.eval = np.sqrt(self.S_ ** 2 + self.lmda ** 2 / 4)
        self.eval_inv = np.diag(1. / self.eval)
        self.eval_extra_dim = self.sign * self.lmda / 2 * np.ones(self.dim_diff)
        self.eval_extra_dim_inv = np.diag(1 / self.eval_extra_dim)

        self.B_inv = np.linalg.pinv(self.B)

        # self.A_0 = S

        self.t = 0
    def forward(self, learning_rate):
        # performs forward for one timestep

        time_step = self.t * learning_rate

        i_in = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim)
        i_hid = np.identity(self.hidden_dim)
        e_eval_st = np.diag(np.exp(1. * self.eval * time_step))
        e_eval_2st = np.diag(np.exp(2. * self.eval * time_step))

        e_eval_st_inv = np.diag(np.exp(-1. * self.eval * time_step))
        e_eval_2st_inv = np.diag(np.exp(-2. * self.eval * time_step))
        e_eval_st_extra_dim = np.diag(np.exp(1. * self.eval_extra_dim * time_step))
        e_eval_2st_extra_dim = np.diag(np.exp(2. * self.eval_extra_dim * time_step))

        # Sinv = np.diag(1./self.A_0)

        if self.U_hat is None and self.V_hat is None:

            O = (1 / np.sqrt(2))*np.vstack([np.hstack([ self.V_ @ (self.A - self.X @ self.A), self.V_ @ (self.A + self.X @ self.A) ]),
                np.hstack([ self.U_ @ (self.A + self.X @ self.A), -self.U_ @ (self.A - self.X @ self.A)])
                ])

            P = np.vstack([
                e_eval_st @self.B.T,
                -  e_eval_st_inv @ self.C.T
            ])
            Z = O @ P
            center_add = 0


        else:
            e_eval_st_extra_dim_2 = np.diag(np.exp((self.sign * self.lmda / 2 - 1. * self.eval) * time_step))
            e_eval_st_extra_dim_2 = np.diag(np.exp((self.sign * self.lmda / 2 - 1. * self.eval) * time_step))

            Z_add =  np.vstack([
                 self.V_hat @ e_eval_st_extra_dim @ self.V_hat.T @ self.V @ self.S1.T ,
                self.U_hat @  e_eval_st_extra_dim @ self.U_hat.T @ self.U @ self.S2
            ])

            # h=Z_add_2-Z_add
            O = (1 / np.sqrt(2)) * np.vstack(
                [np.hstack([self.V_ @ (self.A - self.X @ self.A), self.V_ @ (self.A + self.X @ self.A)]),
                 np.hstack([self.U_ @ (self.A + self.X @ self.A), -self.U_ @ (self.A - self.X @ self.A)])
                 ])

            P = np.vstack([
                e_eval_st @ self.B.T,
                -  e_eval_st_inv @ self.C.T
            ])
            Z =  (1 / np.sqrt(2))  * O @ P
            Z = Z + Z_add

            center_add =   (
                    self.S1 @ self.V.T @ self.V_hat + self.S2.T @ self.U.T @ self.U_hat) @ (
                                e_eval_st_extra_dim - np.eye(self.dim_diff)) @ self.eval_extra_dim_inv @ (
                                    self.V_hat.T @ self.V @ self.S1.T + self.U_hat.T @ self.U @ self.S2)

        center_left =  i_hid

        center_center = 0.25*self.B @ (e_eval_2st - i_in) @ self.eval_inv @ self.B.T

        center_right =  0.25 * self.C @ (
                    e_eval_2st_inv - i_in) @ self.eval_inv @ self.C.T

        center = (center_left + center_center - center_right + center_add )

        qqt = Z @ np.linalg.inv(center) @ Z.T

        # CHOLESKY
        #L = np.linalg.cholesky(center)
        #y = np.linalg.solve(L, Z.T)
        #x = np.linalg.solve(L.T, y)
        #qqt = x.T @ Z.T

        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim]
        self.t= self.t+1

        return qqt


