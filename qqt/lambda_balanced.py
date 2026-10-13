
import numpy as np
from utils import SingularMatrixError




##implement the solution for lambda balanced weights, see if it works


# class SingularMatrixError(Exception):
#     """Exception raised when a matrix is singular."""
#     pass

##method for generating lambda balanced weights goes here

# def reshape_matrix(input_matrix, new_shape):

#     old_shape = input_matrix.shape
    
#     if new_shape[0] > old_shape[0]:

#         new_matrix = np.vstack((input_matrix, np.zeros((new_shape[0] - old_shape[0], old_shape[1]))))
#     elif new_shape[0] < old_shape[0]:

#         new_matrix = input_matrix[:new_shape[0], :]
#     else:
#         new_matrix = input_matrix
    
#     if new_shape[1] > old_shape[1]:
#         new_matrix = np.hstack((new_matrix, np.zeros((new_shape[0], new_shape[1] - old_shape[1]))))
#     elif new_shape[1] < old_shape[1]:
#         new_matrix = new_matrix[:, :new_shape[1]]
    
#     return new_matrix

# def balanced_weights(in_dim, hidden_dim, out_dim):

#     sigma = 1

#     U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
#     r = U @ V.T

#     w1 = np.random.randn(hidden_dim, in_dim)
#     w2 = np.random.randn(out_dim, hidden_dim)

#     U_, S_, V_ = np.linalg.svd(w2 @ w1)
#     s = np.sqrt(np.diag(S_))

#     lmda = np.trace(w2 @ w1) / hidden_dim

#     factor = (- lmda + np.sqrt(lmda ** 2 + 4 * s ** 2)) / 2

#     s_2 = np.sqrt(np.diag(np.diag(factor)))

#     s2_reshaped = reshape_matrix(s_2, (out_dim, hidden_dim))

#     s_1 = np.diag(np.diag(s) / np.diag(s_2))

#     s1_reshaped = reshape_matrix(s_1, (hidden_dim, in_dim))

#     S_test = s2_reshaped @ s1_reshaped

#     w1_out = r @ s1_reshaped @ V_.T 

#     w2_out = U_ @ s2_reshaped @ r.T

#     q = w1_out @ w1_out.T - w2_out.T @ w2_out

#     scale_by = lmda / q[0][0]
#     w1_out = scale_by * w1_out
#     w2_out = scale_by * w2_out
#     q = w1_out @ w1_out.T - w2_out.T @ w2_out

#     return w1_out, w2_out, S_test, q

## class that computes dynamics goes here
#doing for equal input output dimensions (all the assumptions from Fukumizu)
# class QQT_lambda_balanced:
#     def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

#         self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]
        
#         self.weights_only = weights_only
#         self.batch_size = X.shape[0]

#         self.input_dim = X.shape[1]
#         self.output_dim = Y.shape[1]
        
#         sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

#         U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
#         V_ = Vt_.T 

#         self.F = np.vstack([
#         np.hstack([self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
#         np.hstack([sigma_yx_tilde, - self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
#         ]) 

#         self.U_, self.S_, self.V_ = U_, np.diag(S_), V_

#         U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
#         self.U, self.S, self.V = U, S, Vt.T 

#         self.F_inv = np.linalg.inv(self.F)

#         self.O = 1/np.sqrt(2) * np.vstack([
#             np.hstack([self.V_, self.V_]),
#             np.hstack([self.U_, -self.U_])
#         ])

#         self.lmda_big = np.vstack([
#             np.hstack([self.S_, np.zeros((self.S_.shape[0], self.S_.shape[0]))]),
#             np.hstack([np.zeros((self.S_.shape[0], self.S_.shape[0])), -1. * self.S_])
#         ])

#         self.A = np.vstack([
#         np.hstack([self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), np.zeros((sigma_yx_tilde.shape[1], sigma_yx_tilde.shape[1]))]),
#         np.hstack([np.zeros((sigma_yx_tilde.shape[0], sigma_yx_tilde.shape[0])), - self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
#         ]) 

#         self.t = 0



#     def forward(self, learning_rate):
#         #performs forward for one timestep

#         time_step = self.t * learning_rate

#         i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

#         e_st_inv = np.diag(np.exp(-1. * np.diag(self.S_) * time_step))
#         e_2st_inv = np.diag(np.exp(-2. * np.diag(self.S_) * time_step))

#         Sinv = np.diag(1. / self.S)
#         S_inv = np.diag(1. / np.diag(self.S_))

#         B_t = np.exp(-self.lmda * time_step) * self.U.T @ self.U_ + self.V.T @ self.V_
#         C_t = np.exp(-self.lmda * time_step) * self.U.T @ self.U_ - self.V.T @ self.V_
        
#         B_t_inv = np.linalg.inv(B_t)

#         # e_Ft_inv = (e -A) @ O @ (e -lmda) @ O.T 

#         e_Ft_inv = expm(-1. * self.A * time_step) @ self.O @ expm(-1. * self.lmda_big * time_step) @ self.O.T
#         # e_Ft_inv = expm(-1. * self.F * time_step)


#         X_t = self.F_inv - e_Ft_inv @ self.F_inv @ e_Ft_inv

#         e_lmda_inv = np.exp(-self.lmda * time_step)

#         Z = np.vstack([
#                 self.V_ @ (i - e_st_inv @ C_t.T @ B_t_inv.T @ e_st_inv),
#                 self.U_ @ (i + e_st_inv @ C_t.T @ B_t_inv.T @ e_st_inv)
#             ])
        
#         center_left = 4. * e_st_inv @ B_t_inv @ Sinv @ B_t_inv.T @ e_st_inv 

#         center_center = 0.5 * Z.T @ X_t @ Z 

#         center = center_left + center_center 

#         qqt = Z @ np.linalg.inv(center) @ Z.T

#         ##CHOLESKY
#         # L = np.linalg.cholesky(center)
#         # y = np.linalg.solve(L, Z.T)
#         # x = np.linalg.solve(L.T, y)
#         # qqt = x.T @ Z.T

        
#         if self.weights_only:
#             qqt = qqt[self.input_dim:, :self.input_dim] 

#         self.t+=1
#         return qqt 
    

# class QQT_lambda_balanced2:
#     def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

#         self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]

        

        
#         self.weights_only = weights_only
#         self.batch_size = X.shape[0]

#         self.input_dim = X.shape[1]
#         self.output_dim = Y.shape[1]

#         i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 
        
#         sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

#         U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
#         V_ = Vt_.T 

#         self.F = np.vstack([
#         np.hstack([- self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
#         np.hstack([sigma_yx_tilde, self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
#         ]) 

#         self.U_, self.S_, self.V_ = U_, np.diag(S_), V_

#         U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
#         self.U, self.S, self.V = U, S, Vt.T

#         self.S_inv = np.diag(1. / np.diag(self.S_))

#         self.B = U.T @ U_ + Vt @ V_
#         # self.B = U.T @ U_ @ (i - self.lmda/2 * np.diag(self.S_inv)) + Vt @ V_ @ (i + self.lmda/2 * np.diag(self.S_inv))
#         self.C = U.T @ U_ - Vt @ V_

#         if np.isclose(np.linalg.det(self.B), 0):
#             print('init_w1: ', init_w1)
#             print('init_w2: ', init_w2)
#             print('sigma_yx: ', sigma_yx_tilde)
#             print('B: ', self.B)
#             raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")

#         self.B_inv = np.linalg.inv(self.B)
        
#         self.A_0 = S

#         self.t = 0



#     def forward(self, learning_rate):
#         #performs forward for one timestep

#         time_step = self.t * learning_rate

#         i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

#         e_st_inv = np.diag(np.exp(-1. * np.diag(self.S_) * time_step))
#         e_2st_inv = np.diag(np.exp(-2. * np.diag(self.S_) * time_step))

#         S_inv = np.diag(1. / np.diag(self.S_))
#         Sinv = np.diag(1./self.A_0)


#         ##TODO MAYBE S FACTOR IS WRONG
#         # S_factor = np.diag(1/(np.diag(self.S_) + self.lmda**2/4*(1/np.diag(self.S_))))
        
#         S_factor = np.linalg.inv(self.S_ + self.lmda**2 / 4 * self.S_inv)
        
        

#         e_lmdat_inv = np.diag(np.exp(-1. * (np.diag(self.S_) + 1/4 * self.lmda**2 * np.diag(S_inv)) * time_step))
#         e_2lmdat_inv = np.diag(np.exp(-2. * (np.diag(self.S_) + 1/4 * self.lmda**2 * np.diag(S_inv)) * time_step))

#         Z = np.vstack([
#             self.V_ @ ((i + self.lmda/2 *np.diag(S_inv)) - e_st_inv @ self.C.T @ self.B_inv.T @ e_lmdat_inv),
#             self.U_ @ ((i - self.lmda/2 *np.diag(S_inv)) + e_st_inv @ self.C.T @ self.B_inv.T @ e_lmdat_inv),
#         ])


#         center_left = 4 * e_lmdat_inv @ self.B_inv @ Sinv @ self.B_inv.T @ e_lmdat_inv

#         center_center = (i - e_2lmdat_inv) @ S_factor

#         center_right = e_lmdat_inv @ self.B_inv @ self.C @ (e_2st_inv - i) @ S_inv @ self.C.T @ self.B_inv.T @ e_lmdat_inv

#         center = center_left + center_center - center_right

#         qqt = Z @ np.linalg.inv(center) @ Z.T

#         ##CHOLESKY
#         # L = np.linalg.cholesky(center)
#         # y = np.linalg.solve(L, Z.T)
#         # x = np.linalg.solve(L.T, y)
#         # qqt = x.T @ Z.T

        
#         if self.weights_only:
#             qqt = qqt[self.input_dim:, :self.input_dim] 

#         self.t+=1
#         return qqt 

class QQT_lambda_balanced3:
    def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

        self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0] 

        self.init_w1 = init_w1
        self.init_w2  = init_w2
        

        self.weights_only = weights_only
        self.batch_size = X.shape[0]

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]



        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 
        
        # sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 
    
        sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

        U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
        V_ = Vt_.T 

        self.F = np.vstack([
        np.hstack([- self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
        np.hstack([sigma_yx_tilde, self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
        ]) 

        self.U_, self.S_, self.V_ = U_, np.diag(S_), V_
        

        U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
        # U, _, _ = np.linalg.svd(init_w2)
        # _, _, Vt = np.linalg.svd(init_w1)

        self.U, self.S, self.V = U, S, Vt.T

        self.S_inv = np.diag(1. / np.diag(self.S_))

        self.S_ = np.diag(self.S_)

        self.X = (np.sqrt(self.lmda**2 + 4*self.S_**2) - 2 * self.S_)/self.lmda
        self.A = 1 / (np.sqrt(1 + self.X**2))

        self.X = np.diag(self.X)
        self.A = np.diag(self.A)


        self.S2 = np.sqrt((self.lmda + np.sqrt(self.lmda**2 + 4*self.S**2)) / 2)
        self.S1 = self.S / self.S2

        self.S1 = np.diag(self.S1)
        self.S2 = np.diag(self.S2)
        self.B = self.S2 @ U.T @ U_ @ (self.X @ self.A + self.A) + self.S1 @ Vt @ V_ @ (self.A - self.X @ self.A)
        self.C = self.S2 @ U.T @ U_  @ (self.A - self.X @ self.A) - self.S1 @ Vt @ V_ @ (self.X @ self.A + self.A)

        self.eval = np.sqrt(self.S_**2 + self.lmda**2 / 4)
        self.eval_inv = np.diag(1. /self.eval)

        if np.isclose(np.linalg.det(self.B), 0):
            print('init_w1: ', init_w1)
            print('init_w2: ', init_w2)
            print('sigma_yx: ', sigma_yx_tilde)
            print('B: ', self.B)
            raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")

        self.B_inv = np.linalg.inv(self.B)

        
        # self.A_0 = S

        self.t = 0



    def forward(self, learning_rate):
        #performs forward for one timestep

        time_step = self.t * learning_rate

        i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

        e_eval_st_inv  = np.diag(np.exp(-1. * self.eval * time_step))
        e_eval_2st_inv  = np.diag(np.exp(-2. * self.eval * time_step))


        # Sinv = np.diag(1./self.A_0)



        Z = np.vstack([
            self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
            self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
        ])
        
        center_left = 4 * e_eval_st_inv @ self.B_inv @ self.B_inv.T @ e_eval_st_inv

        # center_left = 4 * e_eval_st_inv @ self.B_inv @ Sinv @ self.B_inv.T @ e_eval_st_inv

        center_center = (i - e_eval_2st_inv) @ self.eval_inv

        center_right = e_eval_st_inv @ self.B_inv @ self.C @ (e_eval_2st_inv - i) @ self.eval_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv

        center = center_left + center_center - center_right

        qqt = Z @ np.linalg.inv(center) @ Z.T

        #CHOLESKY
        # L = np.linalg.cholesky(center)
        # y = np.linalg.solve(L, Z.T)
    # x = np.linalg.solve(L.T, y)
        # qqt = x.T @ Z.T

        self.hidden_dim = len(self.init_w1)

        add_term = np.diag([self.lmda for _ in range(self.input_dim)] + [-self.lmda for _ in range(self.input_dim)])

        qqt = qqt + add_term

        if self.weights_only:
            qqt = qqt[self.input_dim:, :self.input_dim] 

        self.t+=1
        return qqt 



# class QQT_lambda_balanced4:
#     def __init__(self, init_w1, init_w2, X, Y, weights_only=False):

#         self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0] 

        

#         self.weights_only = weights_only
#         self.batch_size = X.shape[0]

#         self.input_dim = X.shape[1]
#         self.output_dim = Y.shape[1]



#         i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 
        
#         sigma_yx_tilde = 1 / self.batch_size * Y.T @ X 

#         U_, S_, Vt_= np.linalg.svd(sigma_yx_tilde)
#         V_ = Vt_.T 

#         self.F = np.vstack([
#         np.hstack([- self.lmda / 2 * np.eye(sigma_yx_tilde.shape[1]), sigma_yx_tilde.T]),
#         np.hstack([sigma_yx_tilde, self.lmda / 2 * np.eye(sigma_yx_tilde.shape[0])])
#         ]) 

#         self.U_, self.S_, self.V_ = U_, np.diag(S_), V_

#         self.dim_diff = np.abs(self.input_dim - self.output_dim)

#         if self.input_dim < self.output_dim:
#             U_hat = U_[:, self.input_dim:]
#             V_hat = np.zeros((self.input_dim, self.dim_diff))
#             U_ = U_[:, :self.input_dim]

#         elif self.input_dim > self.output_dim:
#             U_hat = np.zeros((self.output_dim, self.dim_diff))
#             V_hat = V_[:, self.output_dim:]
#             V_ = V_[:, :self.output_dim]
        
#         else:
#             U_hat = None 
#             V_hat = None 

#         self.U_hat = U_hat 
#         self.V_hat = V_hat 
#         self.V_ = V_
#         self.U_ = U_

#         U, S, Vt  = np.linalg.svd(init_w2 @ init_w1, full_matrices=False)
#         self.U, self.S, self.V = U, S, Vt.T

#         self.S_inv = np.diag(1. / np.diag(self.S_))

#         self.S_ = np.diag(self.S_)

#         self.X = (np.sqrt(self.lmda**2 + 4*self.S_**2) - 2 * self.S_)/self.lmda
#         self.A = 1 / (np.sqrt(1 + self.X**2))

#         self.X = np.diag(self.X)
#         self.A = np.diag(self.A)

#         self.B = U.T @ U_ @ (self.X @ self.A + self.A) + Vt @ V_ @ (self.A - self.X @ self.A)
#         self.C = U.T @ U_  @ (self.A - self.X @ self.A) - Vt @ V_ @ (self.X @ self.A + self.A)

#         self.eval = np.sqrt(S_**2 + self.lmda**2 / 4)
#         self.eval_inv = np.diag(1. /self.eval)

#         if np.isclose(np.linalg.det(self.B), 0):
#             print('init_w1: ', init_w1)
#             print('init_w2: ', init_w2)
#             print('sigma_yx: ', sigma_yx_tilde)
#             print('B: ', self.B)
#             raise SingularMatrixError("B IS A SINGULAR MATRIX, CHECK INPUT")

#         self.B_inv = np.linalg.inv(self.B)
        
#         self.A_0 = S

#         self.t = 0



#     def forward(self, learning_rate):
#         #performs forward for one timestep

#         time_step = self.t * learning_rate

#         i = np.identity(self.input_dim) if self.input_dim < self.output_dim else np.identity(self.output_dim) 

#         e_eval_st_inv  =  np.diag(np.exp(-1. * self.eval * time_step))
#         e_eval_2st_inv  = np.diag(np.exp(-2. * self.eval * time_step))


#         Sinv = np.diag(1./self.A_0)


#         if self.U_hat is None and self.V_hat is None: 
#             Z = np.vstack([
#                 self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
#                 self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv),
#             ])
#             center_add = 0.
        
#         else:
#             Z = np.vstack([
#                 self.V_ @ ((self.A - self.X @ self.A) - (self.A + self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv) + 2*self.V_hat@self.V_hat.T @ self.V @ self.B_inv.T @ e_eval_st_inv,
#                 self.U_ @ ((self.A + self.X @ self.A) + (self.A - self.X @ self.A) @ e_eval_st_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv) + 2*self.U_hat@self.U_hat.T @ self.U @ self.B_inv.T @ e_eval_st_inv
#             ])

#             center_add = 4 * time_step * e_eval_st_inv @ self.B_inv @ (self.V.T @ self.V_hat @ self.V_hat.T @ self.V + self.U.T @ self.U_hat @ self.U_hat.T @ self.U) @ self.B_inv.T @ e_eval_st_inv



#         center_left = 4 * e_eval_st_inv @ self.B_inv @ Sinv @ self.B_inv.T @ e_eval_st_inv

#         center_center = (i - e_eval_2st_inv) @ self.eval_inv

#         center_right = e_eval_st_inv @ self.B_inv @ self.C @ (e_eval_2st_inv - i) @ self.eval_inv @ self.C.T @ self.B_inv.T @ e_eval_st_inv

#         center = center_left + center_center - center_right + center_add

#         # qqt = Z @ np.linalg.inv(center) @ Z.T

#         #CHOLESKY
#         L = np.linalg.cholesky(center)
#         y = np.linalg.solve(L, Z.T)
#         x = np.linalg.solve(L.T, y)
#         qqt = x.T @ Z.T

#         self.hidden_dim = len(init_w1)

#         add_term = np.diag([self.lmda for _ in range(self.hidden_dim)] + [-self.lmda for _ in range(self.hidden_dim)])

#         qqt = qqt + add_term
        
#         if self.weights_only:
#             qqt = qqt[self.input_dim:, :self.input_dim] 

#         self.t+=1
#         return qqt 


## plot of dynamics goes here
# in_dim = 5
# hidden_dim = 5
# out_dim = 5

# lmda = 100

# batch_size = 10
# learning_rate = 0.001 / lmda
# training_steps = int(200 * np.sqrt(lmda))

# # init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

# # init_w1, init_w2, _, lmda  = balanced_weights(in_dim, hidden_dim, out_dim)
# # lmda = lmda[0][0]
                         

# b = (np.sqrt(lmda) + 1)
# a = np.sqrt(2*np.sqrt(lmda) + 1)

# init_w1 = np.eye(hidden_dim)*a
# init_w2 = np.eye(hidden_dim)*b

# print("lambda: " ,init_w2.T @ init_w2 - init_w1 @ init_w1.T)


# # init_w1 = np.eye(hidden_dim)*b
# # init_w2 = np.eye(hidden_dim)*a

# # init_w1, init_w2 = zero_balanced_weights(in_dim, hidden_dim, out_dim, 0.35)

# #lets try with lmda = 1


# print((init_w1@init_w1.T - init_w2.T@init_w2)[0][0])


# X, Y = get_random_regression_task(batch_size, in_dim, out_dim)

# U_, S_, Vt_ = np.linalg.svd(Y @ X.T / batch_size)




# ## model solution
# model = LinearNetwork(in_dim, hidden_dim, out_dim, init_w1.copy(), init_w2.copy())
# w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)
# ws = np.array([w2 @ w1 for (w2, w1) in zip(w2s, w1s)])
# ws = np.expand_dims(ws, axis=1)



# # analytical solution 1 (for comparision)
# # analytical = QQT_new(init_w1.copy(), init_w2.copy(), X.T, Y.T, True)
# # analytical = np.asarray([analytical.forward(learning_rate) for _ in range(training_steps)])

# ## analytical solution 2
# analytical2 = QQT_lambda_balanced3(init_w1.copy(), init_w2.copy(), X.T, Y.T, False)
# analytical2 = np.asarray([analytical2.forward(learning_rate) for _ in range(training_steps)])

# # X_ = (np.sqrt(lmda**2 + 4*S_**2) - 2 * S_)/lmda
# # A = 1 / (np.sqrt(1 + X**2))

# # rep1 = [[w1.T @ w1 - lmda * np.eye(hidden_dim)] for w1 in w1s]
# rep1 = [[w1.T @ w1] for w1 in w1s]
# rep1_analytical = np.array([a[:in_dim, :in_dim] for a in analytical2])

# # rep2= [[w2 @ w2.T + lmda * np.eye(hidden_dim)] for w2 in w2s]
# rep2= [[w2 @ w2.T] for w2 in w2s]
# rep2_analytical = np.array([a[in_dim:, in_dim:] for a in analytical2])

# reps = (np.asarray(rep2)[:, 0, :, :])



# # diffs = [np.linalg.norm(a - w) for (a, w) in zip(analytical, analytical2)]
# # print(diffs[:20])
# # print(diffs[-20:])
# # first_a  = [a[0][0] for a in analytical]
# # first_emp = [w[0][0][0] for w in ws]

# # plt.figure()
# # plt.plot(first_a, label = 'analytical')
# # plt.plot(first_emp, label='empirical')
# # plt.legend()
# # plt.show()


# # print(diffs)


# ### PLOTTING TRAJECTORIES OF THE REPRESENTATIONS
# plot_items_n = 4
# blind_colours = BlindColours().get_colours()


# outputs = (np.asarray(ws)[:, 0, :, :] @ X[:,:plot_items_n])

# # representations = (np.asarray(w1s))

# plt.figure()
# reps = (np.asarray(rep2)[:, 0, :, :])
# for color, output in zip(blind_colours, reps.T):
#     for val in output:
#         plt.plot(val, c=color, lw=2.5, label='Representation')
#     plt.plot((rep2_analytical).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
# for color, target in zip(blind_colours, Y[:plot_items_n]):
#     for value in target:
#         plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

# plt.title(f'Representation Dynamics Lambda Balanced, Lambda: {lmda}')
# plt.xlabel('Training Steps')
# plt.ylabel('Network Representation (W2)')
# plt.legend(['output', 'analytical'])


# plt.figure()
# reps = (np.asarray(rep1)[:, 0, :, :])
# for color, output in zip(blind_colours, reps.T):
#     for val in output:
#         plt.plot(val, c=color, lw=2.5, label='Representation')
#     plt.plot((rep1_analytical).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
# for color, target in zip(blind_colours, Y[:plot_items_n]):
#     for value in target:
#         plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

# plt.title(f'Representation Dynamics Lambda Balanced, Lambda: {lmda}')
# plt.xlabel('Training Steps')
# plt.ylabel('Network Representation (W1)')
# plt.legend(['output', 'analytical'])




# plt.figure()

# analytical2 = [a[in_dim:, :in_dim] for a in analytical2]
# for color, output in zip(blind_colours, outputs.T):
#     for val in output:
#         plt.plot(val, c=color, lw=2.5, label='output')
#     plt.plot((analytical2 @ X[:,:plot_items_n]).reshape(training_steps, -1), lw=3., c="k", alpha=0.7, linestyle=(0, (1, 2)), label='analytical') # (0, (3, 4, 3, 1))
    
# for color, target in zip(blind_colours, Y[:plot_items_n]):
#     for value in target:
#         plt.scatter(training_steps * 1.6, value, marker="_", color=color, lw=2.5)

# plt.title(f'Learning Dynamics Lambda Balanced, Lambda: {lmda}')
# plt.xlabel('Training Steps')
# plt.ylabel('Network Output')
# plt.legend(['output', 'analytical'])

# plt.show()