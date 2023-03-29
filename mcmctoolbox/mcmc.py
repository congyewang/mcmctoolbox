import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class MCMC:
    def __init__(self, log_pi, grad_log_pi, nits=5000, theta_start=np.array([-5., -5.]), Sigma=None) -> None:
        self.nits = nits
        self.theta_start = theta_start
        self.log_pi = log_pi
        self.grad_log_pi = grad_log_pi
        self.d = theta_start.size

        if Sigma is None:
            self.Sigma = np.eye(self.d)

        self.store = np.zeros((self.nits+1, self.d))
        self.acc = 0.0

    def K(self, x_array):
        """
        Arbitrary distribution, here the standard two-dimensional Gaussian distribution is used
        """
        x = np.matrix(x_array).T
        res = np.sum(x.T @ x / 2)
        return np.array(res)

    def rwm(self, epsilon=1):
        """
        Random Walk Metropolis-Hastings Algorithm
        """
        nacc = 0
        theta_curr = self.theta_start
        log_pi_curr = self.log_pi(theta_curr)
        self.store[0,:] = theta_curr
        for i in range(self.nits):
            psi = theta_curr + epsilon * np.random.normal(size=self.d)
            log_pi_prop = self.log_pi(psi)
            log_alpha = log_pi_prop - log_pi_curr
            if np.log(np.random.uniform()) < log_alpha:
                theta_curr = psi
                log_pi_curr = log_pi_prop
                nacc += 1

            self.store[i+1,:] = theta_curr

        self.acc = nacc/self.nits

    def hmc(self, delta=0.14, L=20):
        """
        Hamiltonian Monte Carlo Algorithm
        """
        self.store[0,:] = self.theta_start
        nacc = 0

        for t in range(1, self.nits):
            p0 = np.random.randn(self.d)
            pStar = p0 + delta/2*self.grad_log_pi(self.store[t-1,:])
            xStar = self.store[t-1,:] + delta*pStar

            for jL in range(L):
                pStar = pStar + delta*self.grad_log_pi(xStar)
                xStar = xStar + delta*pStar

            pStar = pStar + delta/2*self.grad_log_pi(xStar)

            U0 = -self.log_pi(self.store[t-1,:])
            UStar = -self.log_pi(xStar)

            K0 = self.K(p0)
            KStar = self.K(pStar)

            log_alpha = (U0 + K0) - (UStar + KStar)

            if np.log(np.random.uniform(0,1)) < log_alpha:
                self.store[t,:] = xStar
                nacc += 1
            else:
                self.store[t,:] = self.store[t-1,:]
        self.acc = nacc / self.nits

    def mala(self, epsilon=0.1, *args):
        nacc = 0
        theta_curr = self.theta_start
        log_pi_curr = self.log_pi(theta_curr, *args)
        grad_log_pi_curr = self.grad_log_pi(theta_curr, *args)
        mu_from_theta = theta_curr + epsilon**2/2 * np.dot(self.Sigma, grad_log_pi_curr)

        self.store[0,:] = theta_curr

        for i in range(self.nits):
            psi = multivariate_normal.rvs(mu_from_theta, epsilon**2 * self.Sigma)

            log_pi_prop = self.log_pi(psi, *args)
            grad_log_pi_prop = self.grad_log_pi(psi, *args)
            mu_from_psi = psi + epsilon**2/2 * np.dot(self.Sigma, grad_log_pi_prop)

            lq_theta_to_psi = multivariate_normal.logpdf(psi, mu_from_theta, epsilon**2 * self.Sigma)
            lq_psi_to_theta = multivariate_normal.logpdf(theta_curr, mu_from_psi, epsilon**2 * self.Sigma)

            log_alpha = log_pi_prop + lq_psi_to_theta - log_pi_curr - lq_theta_to_psi
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                theta_curr = psi
                log_pi_curr = log_pi_prop
                grad_log_pi_curr = grad_log_pi_prop
                mu_from_theta = mu_from_psi
                nacc += 1
            self.store[i+1,:] = theta_curr
        self.acc = nacc / self.nits

    def tmala(self, epsilon=0.01):
        """
        Tamed Metropolis-Adjusted Langevin Algorithm
        """
        nacc = 0
        x = self.theta_start

        taming = lambda g, epsilon: g/(1. + epsilon*np.linalg.norm(g))

        for i in range(self.nits):
            self.store[i,:] = x
            U_x, grad_U_x = -self.log_pi(x), -self.grad_log_pi(x)
            tamed_gUx = taming(grad_U_x, epsilon)
            y = x - epsilon * tamed_gUx + np.sqrt(2*epsilon) * np.random.normal(size=self.d)
            U_y, grad_U_y = -self.log_pi(y), -self.grad_log_pi(y)
            tamed_gUy = taming(grad_U_y, epsilon)

            log_alpha = -U_y + U_x + 1./(4*epsilon) * (np.linalg.norm(y - x + epsilon*tamed_gUx)**2 - np.linalg.norm(x - y + epsilon*tamed_gUy)**2)
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                x = y
                nacc += 1
        self.acc = nacc / self.nits


    def tmalac(self, epsilon=0.01):
        """
        Tamed Metropolis-Adjusted Langevin Algorithm Coordinatewise
        """
        nacc = 0 # acceptance probability
        x = self.theta_start

        taming = lambda g, step: np.divide(g, 1. + step * np.absolute(g))

        for i in range(self.nits):
            self.store[i,:] = x
            U_x, grad_U_x = -self.log_pi(x), -self.grad_log_pi(x)
            tamed_gUx = taming(grad_U_x, epsilon)
            y = x - epsilon * tamed_gUx + np.sqrt(2*epsilon) * np.random.normal(size=self.d)
            U_y, grad_U_y = -self.log_pi(y), -self.grad_log_pi(y)
            tamed_gUy = taming(grad_U_y, epsilon)

            log_alpha = -U_y + U_x + 1./(4*epsilon) * (np.linalg.norm(y - x + epsilon*tamed_gUx)**2 - np.linalg.norm(x - y + epsilon*tamed_gUy)**2)
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                x = y
                nacc += 1
        self.acc = nacc / self.nits

    def q_invariant_mala(self, grad_log_kp, epsilon=1.0):
        """
        Q-Invariant MALA
        """
        nacc = 0
        theta_curr = self.theta_start
        self.store[0,:] = theta_curr

        m = lambda x: epsilon * self.grad_log_pi(x) + (epsilon/2) * grad_log_kp(x)

        for i in range(self.nits):
            theta_star = theta_curr + m(theta_curr) + np.sqrt(2 * epsilon) * np.random.normal(size=self.d)
            log_alpha = self.log_pi(theta_star) - self.log_pi(theta_curr) + 0.5 * (grad_log_kp(theta_star) - grad_log_kp(theta_curr)) - (1/(4*epsilon)) * (np.linalg.norm(theta_curr - m(theta_star), 2)) + (1/(4*epsilon)) * (np.linalg.norm(theta_star - m(theta_curr), 2))
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                theta_curr = theta_star
                nacc += 1
            else:
                theta_curr = theta_curr
            self.store[i+1] = theta_curr
        self.acc = nacc / self.nits

    def plot(self, density=True):
        if self.d == 1:
            plt.plot(self.store, color='black', linewidth=0.7, alpha=0.2, marker='.', linestyle='solid')
            plt.show()

            if density:
                num_bins = 50
                plt.hist(self.store, num_bins, stacked=True, edgecolor="white", alpha=0.5)
                plt.show()

        elif self.d == 2:
            plt.plot(self.store[:,0], self.store[:,1], color='black', linewidth=0.7, alpha=0.2, marker='.', linestyle='solid')
            plt.show()

            if density:
                num_bins = 50
                plt.hist(self.store[:,0], num_bins, stacked=True, edgecolor="white", facecolor='green', alpha=0.5)
                plt.hist(self.store[:,1], num_bins, stacked=True, edgecolor="white", facecolor='red', alpha=0.5)
                plt.show()

        else:
            raise ValueError("Larger than 2D")
