import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal


class MCMC:
    def __init__(
        self,
        log_pi,
        grad_log_pi,
        nits=5000,
        theta_start=np.array([-5.0, -5.0]),
        Sigma=None,
    ) -> None:
        self.nits = nits
        self.theta_start = theta_start
        self.log_pi = log_pi
        self.grad_log_pi = grad_log_pi
        self.d = theta_start.size

        if Sigma is None:
            Sigma = np.eye(self.d)
        self.Sigma = Sigma

        self.store = np.zeros((self.nits + 1, self.d))
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
        self.store[0, :] = theta_curr
        for i in range(self.nits):
            psi = theta_curr + epsilon * np.random.normal(size=self.d)
            log_pi_prop = self.log_pi(psi)
            log_alpha = log_pi_prop - log_pi_curr
            if np.log(np.random.uniform()) < log_alpha:
                theta_curr = psi
                log_pi_curr = log_pi_prop
                nacc += 1

            self.store[i + 1, :] = theta_curr

        self.acc = nacc / self.nits

    def hmc(self, delta=0.14, L=20):
        """
        Hamiltonian Monte Carlo Algorithm
        """
        self.store[0, :] = self.theta_start
        nacc = 0

        for t in range(1, self.nits):
            p0 = np.random.randn(self.d)
            pStar = p0 + delta / 2 * self.grad_log_pi(self.store[t - 1, :])
            xStar = self.store[t - 1, :] + delta * pStar

            for jL in range(L):
                pStar = pStar + delta * self.grad_log_pi(xStar)
                xStar = xStar + delta * pStar

            pStar = pStar + delta / 2 * self.grad_log_pi(xStar)

            U0 = -self.log_pi(self.store[t - 1, :])
            UStar = -self.log_pi(xStar)

            K0 = self.K(p0)
            KStar = self.K(pStar)

            log_alpha = (U0 + K0) - (UStar + KStar)

            if np.log(np.random.uniform(0, 1)) < log_alpha:
                self.store[t, :] = xStar
                nacc += 1
            else:
                self.store[t, :] = self.store[t - 1, :]
        self.acc = nacc / self.nits

    def mala(self, epsilon=0.1, *args):
        nacc = 0
        theta_curr = self.theta_start
        log_pi_curr = self.log_pi(theta_curr, *args)
        grad_log_pi_curr = self.grad_log_pi(theta_curr, *args)
        mu_from_theta = theta_curr + epsilon**2 / 2 * np.dot(
            self.Sigma, grad_log_pi_curr
        )

        self.store[0, :] = theta_curr

        for i in range(self.nits):
            psi = multivariate_normal.rvs(mu_from_theta, epsilon**2 * self.Sigma)

            log_pi_prop = self.log_pi(psi, *args)
            grad_log_pi_prop = self.grad_log_pi(psi, *args)
            mu_from_psi = psi + epsilon**2 / 2 * np.dot(self.Sigma, grad_log_pi_prop)

            lq_theta_to_psi = multivariate_normal.logpdf(
                psi, mu_from_theta, epsilon**2 * self.Sigma
            )
            lq_psi_to_theta = multivariate_normal.logpdf(
                theta_curr, mu_from_psi, epsilon**2 * self.Sigma
            )

            log_alpha = log_pi_prop + lq_psi_to_theta - log_pi_curr - lq_theta_to_psi
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                theta_curr = psi
                log_pi_curr = log_pi_prop
                grad_log_pi_curr = grad_log_pi_prop
                mu_from_theta = mu_from_psi
                nacc += 1
            self.store[i + 1, :] = theta_curr
        self.acc = nacc / self.nits

    def tmala(self, epsilon=0.01):
        """
        Tamed Metropolis-Adjusted Langevin Algorithm
        """
        nacc = 0
        x = self.theta_start

        taming = lambda g, epsilon: g / (1.0 + epsilon * np.linalg.norm(g))

        for i in range(self.nits):
            self.store[i, :] = x
            U_x, grad_U_x = -self.log_pi(x), -self.grad_log_pi(x)
            tamed_gUx = taming(grad_U_x, epsilon)
            y = (
                x
                - epsilon * tamed_gUx
                + np.sqrt(2 * epsilon) * np.random.normal(size=self.d)
            )
            U_y, grad_U_y = -self.log_pi(y), -self.grad_log_pi(y)
            tamed_gUy = taming(grad_U_y, epsilon)

            log_alpha = (
                -U_y
                + U_x
                + 1.0
                / (4 * epsilon)
                * (
                    np.linalg.norm(y - x + epsilon * tamed_gUx) ** 2
                    - np.linalg.norm(x - y + epsilon * tamed_gUy) ** 2
                )
            )
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                x = y
                nacc += 1
        self.acc = nacc / self.nits

    def tmalac(self, epsilon=0.01):
        """
        Tamed Metropolis-Adjusted Langevin Algorithm Coordinatewise
        """
        nacc = 0  # acceptance probability
        x = self.theta_start

        taming = lambda g, step: np.divide(g, 1.0 + step * np.absolute(g))

        for i in range(self.nits):
            self.store[i, :] = x
            U_x, grad_U_x = -self.log_pi(x), -self.grad_log_pi(x)
            tamed_gUx = taming(grad_U_x, epsilon)
            y = (
                x
                - epsilon * tamed_gUx
                + np.sqrt(2 * epsilon) * np.random.normal(size=self.d)
            )
            U_y, grad_U_y = -self.log_pi(y), -self.grad_log_pi(y)
            tamed_gUy = taming(grad_U_y, epsilon)

            log_alpha = (
                -U_y
                + U_x
                + 1.0
                / (4 * epsilon)
                * (
                    np.linalg.norm(y - x + epsilon * tamed_gUx) ** 2
                    - np.linalg.norm(x - y + epsilon * tamed_gUy) ** 2
                )
            )
            if np.log(np.random.uniform(0, 1)) < log_alpha:
                x = y
                nacc += 1
        self.acc = nacc / self.nits

    def simulated_annealing_mcmc(
        self, initial_temp=10, cooling_rate=0.999, epsilon=1.0
    ):
        """
        Simulated Annealing MH
        """
        nacc = 0
        theta_curr = self.theta_start
        log_pi_curr = self.log_pi(theta_curr)
        self.store[0, :] = theta_curr
        current_temp = initial_temp

        for i in range(self.nits):
            psi = theta_curr + epsilon * np.random.normal(size=self.d)
            log_pi_prop = self.log_pi(psi)
            log_alpha = min(0, (log_pi_prop - log_pi_curr) / current_temp)
            if np.log(np.random.uniform()) < log_alpha:
                theta_curr = psi
                log_pi_curr = log_pi_prop
                nacc += 1

            self.store[i + 1, :] = theta_curr
            current_temp *= cooling_rate

        self.acc = nacc / self.nits

    def fisher_adaptive_mala(self, lambda_=10, alpha_star=0.574, n0=500, rho_n=0.015):
        """
        Fisher adaptive MALA algorithm

        Reference:
            Titsias, M. K. (2023). Optimal Preconditioning and Fisher Adaptive Langevin Sampling. arXiv preprint arXiv:2305.14442.
        """

        def h(z, v, sigma2, grad_log_pi_v):
            """
            Define the function h(z, v) used in Proposition 1
            """
            diff = z - v - (sigma2 / 4) * grad_log_pi_v
            return 0.5 * np.dot(diff, grad_log_pi_v)

        def compute_R_n(s_n, R_prev):
            """
            Define the function to compute the square root matrix R_n used in Proposition 2
            """
            phi_n = np.dot(R_prev.T, s_n)
            r_n = 1 / (1 + np.sqrt(1 / (1 + np.dot(phi_n, phi_n))))
            R_n = R_prev - r_n * np.outer(np.dot(R_prev, phi_n), phi_n) / (
                1 + np.dot(phi_n, phi_n)
            )
            return R_n

        # Initialize theta_curr and sigma2
        theta_curr = self.theta_start
        sigma2 = 1.0

        # Run simple MALA to initialize theta_curr and sigma2
        for _ in range(n0):
            eta = np.random.normal(size=2)
            x_prime = (
                theta_curr
                + (sigma2 / 2) * self.grad_log_pi(theta_curr)
                + np.sqrt(sigma2) * eta
            )
            log_pi_x_prime = self.log_pi(x_prime)
            alpha = min(1, np.exp(log_pi_x_prime - self.log_pi(theta_curr)))
            if np.random.rand() < alpha:
                theta_curr = x_prime
                sigma2 = sigma2 * (1 + rho_n * (alpha - alpha_star))

        # Initialize R, sigma_R2, and compute log_pi and grad_log_pi for theta_curr
        R = np.eye(self.d)
        sigma_R2 = sigma2
        log_pi_curr, grad_log_pi_curr = self.log_pi(theta_curr), self.grad_log_pi(
            theta_curr
        )

        nacc = 0

        # Main loop
        for i in range(self.nits):
            # Propose theta_prop
            eta = np.random.normal(size=self.d)
            theta_prop = (
                theta_curr
                + (sigma_R2 / 2) * np.dot(R, np.dot(R.T, grad_log_pi_curr))
                + np.sqrt(sigma_R2) * np.dot(R, eta)
            )
            log_pi_prop, grad_log_pi_prop = self.log_pi(theta_prop), self.grad_log_pi(
                theta_prop
            )

            # Compute acceptance probability alpha(theta_curr, theta_prop)
            h_curr_prop = h(theta_curr, theta_prop, sigma_R2, grad_log_pi_prop)
            h_prop_curr = h(theta_prop, theta_curr, sigma_R2, grad_log_pi_curr)
            alpha = min(
                1, np.exp(log_pi_prop + h_curr_prop - log_pi_curr - h_prop_curr)
            )

            # Adapt R and sigma2
            s_n_delta = np.sqrt(alpha) * (grad_log_pi_prop - grad_log_pi_curr)
            if i == 0:
                R = compute_R_n(s_n_delta, R / np.sqrt(lambda_))
            else:
                R = compute_R_n(s_n_delta, R)
            sigma2 = sigma2 * (1 + rho_n * (alpha - alpha_star))
            sigma_R2 = sigma2 / (np.trace(np.dot(R, R.T)) / self.d)

            # Accept/reject
            if np.random.rand() < alpha:
                theta_curr = theta_prop
                log_pi_curr, grad_log_pi_curr = log_pi_prop, grad_log_pi_prop
                nacc += 1

            self.store[i] = theta_curr
        self.acc = nacc / self.nits

    def am_mh(
        self, initial_mu=None, lambda_initial=1, gamma_sequence=None, alpha_star=0.234
    ):
        """
        AM algorithm with global adaptive scaling.
        """
        if initial_mu is None:
            initial_mu = np.repeat(0.0, self.d)
        if gamma_sequence is None:
            gamma_sequence = [1 / (i + 1) ** 0.6 for i in range(self.nits)]

        # Initialize
        theta = np.zeros((self.nits + 1, self.d))
        mu = np.zeros((self.nits + 1, self.d))
        Sigma = np.zeros((self.nits + 1, self.d, self.d))
        lambda_seq = np.zeros(self.nits + 1)

        theta[0] = self.theta_start
        mu[0] = initial_mu
        Sigma[0] = self.Sigma
        lambda_seq[0] = lambda_initial

        nacc = 0

        for i in range(self.nits):
            # Sample from proposal distribution
            theta_prop = np.random.multivariate_normal(
                theta[i], lambda_seq[i] * Sigma[i]
            )
            log_alpha = min(0.0, self.log_pi(theta_prop) - self.log_pi(theta[i]))

            # Accept or reject
            if np.log(np.random.uniform()) < log_alpha:
                theta[i + 1] = theta_prop
                nacc += 1
            else:
                theta[i + 1] = theta[i]

            # Update parameters
            lambda_seq[i + 1] = np.exp(
                np.log(lambda_seq[i])
                + gamma_sequence[i] * (np.exp(log_alpha) - alpha_star)
            )
            mu[i + 1] = mu[i] + gamma_sequence[i] * (theta[i + 1] - mu[i])
            Sigma[i + 1] = Sigma[i] + gamma_sequence[i] * (
                np.outer(theta[i + 1] - mu[i], theta[i + 1] - mu[i]) - Sigma[i]
            )

        self.store = theta[1:]
        self.acc = nacc / self.nits

    def plot(self, num_bins=50):
        if self.d == 1:
            _, ax = plt.subplots(1, 2)
            ax[0].plot(
                self.store[:, 0],
                color="black",
                linewidth=0.7,
                alpha=0.2,
                marker=".",
                linestyle="solid",
            )
            ax[1].hist(
                self.store[:, 0],
                num_bins,
                stacked=True,
                edgecolor="white",
                facecolor="red",
                alpha=0.5,
            )
            plt.show()
        elif self.d == 2:
            g = sns.jointplot(
                x=self.store[:, 0], y=self.store[:, 1], linewidth=0.7, alpha=0.2
            )
            g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
            g.plot_marginals(sns.rugplot, color="r", height=-0.15, clip_on=False)
            plt.show()
        else:
            _, ax = plt.subplots(self.d, 2)
            for i in range(self.d):
                ax[i, 0].plot(
                    self.store[:, i],
                    color="black",
                    linewidth=0.7,
                    alpha=0.2,
                    marker=".",
                    linestyle="solid",
                )
                ax[i, 1].hist(
                    self.store[:, i],
                    num_bins,
                    stacked=True,
                    edgecolor="white",
                    facecolor="red",
                    alpha=0.5,
                )
            plt.show()
