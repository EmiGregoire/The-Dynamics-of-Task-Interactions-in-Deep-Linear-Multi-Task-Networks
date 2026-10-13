import numpy as np 



class Aligned_Dynamics:
    def __init__(self, init_w1, init_w2, X, Y):
        self.lmda = (init_w1 @ init_w1.T - init_w2.T @ init_w2)[0][0]

        sigma_yx_tilde = 1/ X.shape[1] * Y @ X.T

        self.U, self.S, self.Vt = np.linalg.svd(sigma_yx_tilde)

        self.a0 = np.diag(self.U.T @ init_w2 @ init_w1 @ self.Vt.T)

        self.theta0 = np.arcsinh(2/self.lmda * np.diag(self.U.T @ init_w2 @ init_w1 @ self.Vt.T))

        self.K = np.sqrt(self.lmda**2 + 4 * self.S**2)

        term = np.sign(self.lmda) * 2 * self.a0 / (np.sqrt(self.lmda**2+4*self.a0**2) + np.abs(self.lmda))

        self.c0 = (self.K + self.lmda + 2*self.S*term) / (self.K - self.lmda - 2 * self.S * term)

        # self.c02 = (self.K + self.lmda + 2*self.S*np.tanh(self.theta0/2)) / (self.K - self.lmda - 2*self.S*np.tanh(self.theta0/2))
        
        self.t = 0
        
        
    def forward(self, learning_rate):
        time_step = self.t * learning_rate

        K_exp = np.exp(self.K * time_step * np.sign(self.lmda))

        numerator = self.K * (self.c0 * K_exp - 1) - self.lmda * (self.c0 * K_exp+1)
        denominator = 2 * self.S * (self.c0 * K_exp + 1)

        # numerator = self.K * (self.c02 * K_exp - 1) - self.lmda * (self.c02 * K_exp+1)
        # denominator = 2 * self.S * (self.c02 * K_exp + 1)

        # theta_t = 2 * np.arctanh(numerator / denominator)
        x = numerator / denominator 

        a_t = self.lmda * x / (1 - x**2)

        self.t += 1 
        # return theta_t

        return a_t 

