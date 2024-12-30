from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import typing as npt
from scipy.stats import multivariate_normal
from tqdm.auto import trange


class MCMCAlgorithmBase(ABC):
    def __init__(
        self,
        log_target_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        init_sample: npt.NDArray[np.floating],
        covariance: Optional[npt.NDArray[np.floating]] = None,
        nits: int = 5_000,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MCMC object. The object can be used to run various MCMC algorithms.

        Args:
            log_target_pdf (Callable[[T], T]): Logarithm of the target distribution.
            init_sample (npt.NDArray[np.floating]): Starting point of the chain.
            covariance (Optional[npt.NDArray[np.floating]], optional): Covariance matrix of the proposal distribution. Defaults to None.
            nits (int, optional): Number of iterations. Defaults to 5000.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
        """
        self.log_target_pdf = log_target_pdf
        self.init_sample = init_sample
        self.nits = nits
        self.sample_dim = len(init_sample)
        self.verbose = verbose

        if covariance is None:
            covariance = np.eye(self.sample_dim)
        self.covariance = covariance

        self.store = np.zeros((nits + 1, self.sample_dim))
        self.acc = 0.0

    def log_proposal_pdf(
        self,
        x: npt.NDArray[np.floating],
        mean: npt.NDArray[np.floating],
        covariance: npt.NDArray[np.floating],
    ) -> np.floating:
        """
        Logarithm of the proposal distribution.

        Args:
            x (npt.NDArray[np.floating]): Array of values.
            mean (npt.NDArray[np.floating]): Mean of the distribution.
            covariance (npt.NDArray[np.floating]): Covariance matrix of the distribution.

        Returns:
            np.floating: Value of the distribution.
        """
        return multivariate_normal.logpdf(x, mean=mean, cov=covariance)

    @abstractmethod
    def sample(self) -> None:
        """
        Sample from the target distribution.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            npt.NDArray[np.floating]: _description_
        """
        raise NotImplementedError("sample method is not implemented")

    def plot(self, num_bins: int = 50):
        match self.sample_dim:
            case 1:
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
            case 2:
                g = sns.jointplot(
                    x=self.store[:, 0], y=self.store[:, 1], linewidth=0.7, alpha=0.2
                )
                g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
                g.plot_marginals(sns.rugplot, color="r", height=-0.15, clip_on=False)
                plt.show()
            case _:
                _, ax = plt.subplots(self.sample_dim, 2)
                for i in range(self.sample_dim):
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


class RandomWalkMetropolisHastings(MCMCAlgorithmBase):
    def sample(self, epsilon: float = 1) -> None:
        """
        Random Walk Metropolis-Hastings Algorithm with Gaussian proposal distribution.

        Args:
            epsilon (float, optional): Standard deviation of the proposal distribution. Defaults to 1.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
        """
        nacc = 0
        current_sample = self.init_sample
        log_target_current = self.log_target_pdf(current_sample)

        self.store[0, :] = current_sample

        for i in trange(self.nits, disable=not self.verbose):
            proposed_sample = current_sample + epsilon * np.random.normal(
                size=self.sample_dim
            )
            log_target_proposed = self.log_target_pdf(proposed_sample)

            log_ratio = log_target_proposed - log_target_current
            log_alpha = np.minimum(0, log_ratio)

            if np.log(np.random.uniform()) < log_alpha:
                current_sample = proposed_sample
                log_target_current = log_target_proposed
                nacc += 1

            self.store[i + 1, :] = current_sample

        self.acc = nacc / self.nits


class AdaptiveMetropolisHastings(MCMCAlgorithmBase):
    def sample(
        self,
        initial_mu: Optional[npt.NDArray[np.floating]] = None,
        lambda_initial: float = 1.0,
        gamma_sequence: Optional[List[float]] = None,
        alpha_star: float = 0.234,
    ) -> None:
        """
        AM algorithm with global adaptive scaling.

        Args:
            initial_mu (Optional[npt.NDArray[np.floating]], optional): Initial mean of the proposal distribution. Defaults to None.
            lambda_initial (float, optional): Initial scaling factor. Defaults to 1.0.
            gamma_sequence (Optional[List[float]], optional): Sequence of scaling factors. Defaults to None.
            alpha_star (float, optional): Target acceptance rate. Defaults to 0.234.
        """
        if initial_mu is None:
            initial_mu = np.repeat(0.0, self.sample_dim)
        if gamma_sequence is None:
            gamma_sequence = [1 / (i + 1) ** 0.6 for i in range(self.nits)]

        # Initialize
        samples = np.zeros((self.nits + 1, self.sample_dim))
        means = np.zeros((self.nits + 1, self.sample_dim))
        covariances = np.zeros((self.nits + 1, self.sample_dim, self.sample_dim))
        lambda_seq = np.zeros(self.nits + 1)

        samples[0] = self.init_sample
        means[0] = initial_mu
        covariances[0] = self.covariance
        lambda_seq[0] = lambda_initial

        nacc = 0

        for i in trange(self.nits, disable=not self.verbose):
            proposed_sample = np.random.multivariate_normal(
                samples[i], lambda_seq[i] * covariances[i]
            )
            log_ratio = self.log_target_pdf(proposed_sample) - self.log_target_pdf(
                samples[i]
            )
            log_alpha = np.minimum(0.0, log_ratio)

            # Accept or reject
            if np.log(np.random.uniform()) < log_alpha:
                # theta[i + 1] = theta_prop
                samples[i + 1] = proposed_sample
                nacc += 1
            else:
                samples[i + 1] = samples[i]

            # Update parameters
            lambda_seq[i + 1] = np.exp(
                np.log(lambda_seq[i])
                + gamma_sequence[i] * (np.exp(log_alpha) - alpha_star)
            )
            means[i + 1] = means[i] + gamma_sequence[i] * (samples[i + 1] - means[i])
            covariances[i + 1] = covariances[i] + gamma_sequence[i] * (
                np.outer(samples[i + 1] - means[i], samples[i + 1] - means[i])
                - covariances[i]
            )

        self.store = samples[1:]
        self.acc = nacc / self.nits


class MetropolisAdjustedLangevinAlgorithm(MCMCAlgorithmBase):
    def __init__(
        self,
        log_target_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        grad_log_target_pdf: Callable[
            [npt.NDArray[np.floating]], npt.NDArray[np.floating]
        ],
        init_sample: npt.NDArray[np.floating],
        covariance: Optional[npt.NDArray[np.floating]] = None,
        nits: int = 5_000,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MCMC object. The object can be used to run various MCMC algorithms.

        Args:
            log_target_pdf (Callable[[T], T]): Logarithm of the target distribution.
            grad_log_target_pdf (Callable[[T], T]): Gradient of the logarithm of the target distribution.
            init_sample (npt.NDArray[np.floating]): Starting point of the chain.
            covariance (Optional[npt.NDArray[np.floating]], optional): Covariance matrix of the proposal distribution. Defaults to None.
            nits (int, optional): Number of iterations. Defaults to 5000.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
        """
        super().__init__(log_target_pdf, init_sample, covariance, nits, verbose)

        self.grad_log_target_pdf = grad_log_target_pdf

    def sample(self, epsilon: float = 0.1) -> None:
        """
        Metropolis-Adjusted Langevin Algorithm (MALA).

        Args:
            epsilon (float, optional): Step size. Defaults to 0.1.
        """
        nacc = 0
        current_sample = self.init_sample
        log_target_current = self.log_target_pdf(current_sample)
        grad_log_target_current = self.grad_log_target_pdf(current_sample)
        current_mean = current_sample + epsilon**2 / 2 * np.dot(
            self.covariance, grad_log_target_current
        )

        self.store[0, :] = current_sample

        for i in trange(self.nits, disable=not self.verbose):
            proposed_sample = multivariate_normal.rvs(
                current_mean, epsilon**2 * self.covariance
            )

            log_target_proposed = self.log_target_pdf(proposed_sample)
            grad_log_target_proposed = self.grad_log_target_pdf(proposed_sample)
            proposed_mean = proposed_sample + epsilon**2 / 2 * np.dot(
                self.covariance, grad_log_target_proposed
            )

            log_proposal_proposed = self.log_proposal_pdf(
                proposed_sample, current_mean, epsilon**2 * self.covariance
            )
            log_proposal_current = self.log_proposal_pdf(
                current_sample, proposed_mean, epsilon**2 * self.covariance
            )

            log_ratio = (
                log_target_proposed
                + log_proposal_current
                - log_target_current
                - log_proposal_proposed
            )
            log_alpha = np.minimum(0, log_ratio)
            if np.log(np.random.uniform()) < log_alpha:
                current_sample = proposed_sample
                log_target_current = log_target_proposed
                grad_log_target_current = grad_log_target_proposed
                current_mean = proposed_mean
                nacc += 1

            self.store[i + 1, :] = current_sample

        self.acc = nacc / self.nits


class TamedMetropolisAdjustedLangevinAlgorithm(MetropolisAdjustedLangevinAlgorithm):
    def taming(
        self, gradient: npt.NDArray[np.floating], epsilon: float
    ) -> npt.NDArray[np.floating]:
        """
        Taming function.

        Args:
            gradient (npt.NDArray[np.floating]): Gradient.
            epsilon (float): Step size.

        Returns:
            npt.NDArray[np.floating]: Tamed gradient.
        """
        return gradient / (1.0 + epsilon * np.linalg.norm(gradient))

    def sample(self, epsilon: float = 0.01) -> None:
        """
        Tamed Metropolis-Adjusted Langevin Algorithm (tMALA).

        Args:
            epsilon (float, optional): Step size. Defaults to 0.01.
        """
        nacc = 0
        current_sample = self.init_sample

        for i in trange(self.nits, disable=not self.verbose):
            self.store[i, :] = current_sample
            log_target_current = self.log_target_pdf(current_sample)
            grad_log_target_current = self.grad_log_target_pdf(current_sample)

            tamed_grad_log_target_current = self.taming(
                grad_log_target_current, epsilon
            )
            proposed_sample = (
                current_sample
                - epsilon * tamed_grad_log_target_current
                + np.sqrt(2 * epsilon) * np.random.normal(size=self.sample_dim)
            )
            log_target_proposed = self.log_target_pdf(proposed_sample)
            grad_log_target_proposed = self.grad_log_target_pdf(proposed_sample)
            tamed_grad_log_target_proposed = self.taming(
                grad_log_target_proposed, epsilon
            )

            log_ratio = (
                log_target_proposed
                - log_target_current
                + 1.0
                / (4 * epsilon)
                * (
                    np.linalg.norm(
                        proposed_sample
                        - current_sample
                        - epsilon * tamed_grad_log_target_current
                    )
                    ** 2
                    - np.linalg.norm(
                        current_sample
                        - proposed_sample
                        - epsilon * tamed_grad_log_target_proposed
                    )
                    ** 2
                )
            )
            log_alpha = np.minimum(0, log_ratio)
            if np.log(np.random.uniform()) < log_alpha:
                current_sample = proposed_sample
                nacc += 1

        self.acc = nacc / self.nits


class TamedMetropolisAdjustedLangevinAlgorithmCoordinatewise(
    TamedMetropolisAdjustedLangevinAlgorithm
):
    def taming(
        self, gradient: npt.NDArray[np.floating], epsilon: float
    ) -> npt.NDArray[np.floating]:
        """
        Taming function.

        Args:
            gradient (npt.NDArray[np.floating]): Gradient.
            epsilon (float): Step size.

        Returns:
            npt.NDArray[np.floating]: Tamed gradient.
        """
        return np.divide(gradient, 1.0 + epsilon * np.absolute(gradient))


class FisherAdaptiveLangevinMetropolisHastings(MetropolisAdjustedLangevinAlgorithm):
    def proposal_log_correction(
        self,
        x: npt.NDArray[np.floating],
        mean: npt.NDArray[np.floating],
        sigma2: npt.NDArray[np.floating],
        grad_log_target_mean: npt.NDArray[np.floating],
    ) -> np.floating:
        """
        Define the function log correction term used in Proposition 1.

        Args:
            x (npt.NDArray[np.floating]): Sample.
            mean (npt.NDArray[np.floating]): Mean.
            sigma2 (npt.NDArray[np.floating]): Square of Step Size in Covariance Matrix.
            grad_log_target_mean (npt.NDArray[np.floating]): Gradient.

        Returns:
            np.floating: Log correction term.
        """
        diff = x - mean - (sigma2 / 4) * grad_log_target_mean
        return 0.5 * np.dot(diff, grad_log_target_mean)

    def fisher_precondition_update(
        self,
        s_n: npt.NDArray[np.floating],
        R_prev: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """
        Define the function to compute the square root matrix R_n used in Proposition 2.

        Args:
            s_n (npt.NDArray[np.floating]): Gradient difference.
            R_prev (npt.NDArray[np.floating]): Previous square root matrix.

        Returns:
            npt.NDArray[np.floating]: Updated square root matrix.
        """
        phi_n = np.dot(R_prev.T, s_n)
        r_n = 1 / (1 + np.sqrt(1 / (1 + np.dot(phi_n, phi_n))))
        R_n = R_prev - r_n * np.outer(np.dot(R_prev, phi_n), phi_n) / (
            1 + np.dot(phi_n, phi_n)
        )
        return R_n

    def sample(
        self,
        lambda_: float = 10.0,
        alpha_star: float = 0.574,
        n0: int = 500,
        epsilon: float = 0.015,
    ) -> None:
        """
        Fisher adaptive MALA algorithm (FAMALA) with optimal preconditioning.

        Args:
            lambda_ (float, optional): Parameter for optimal preconditioning. Defauts to 10.
            alpha_star (float, optional): Target acceptance rate. Defaults to 0.574.
            n0 (int, optional): Number of iterations for initialization. Defaults to 500.
            epsilon (float, optional): Step size for the adaptation of sigma2. Defaults to 0.015.

        Reference:
            Titsias, M. K. (2023). Optimal Preconditioning and Fisher Adaptive Langevin Sampling. arXiv preprint arXiv:2305.14442.
        """
        nacc = 0
        current_sample = self.init_sample
        sigma2 = 1.0

        # Run simple MALA to initialize theta_curr and sigma2
        for _ in range(n0):
            eta = np.random.normal(size=self.sample_dim)
            current_sample_prime = (
                current_sample
                + (sigma2 / 2) * self.grad_log_target_pdf(current_sample)
                + np.sqrt(sigma2) * eta
            )
            log_target_current_prime = self.log_target_pdf(current_sample_prime)
            log_ratio_mala = log_target_current_prime - self.log_target_pdf(
                current_sample
            )
            log_alpha_mala = np.minimum(0, log_ratio_mala)

            if np.log(np.random.rand()) < log_alpha_mala:
                current_sample = current_sample_prime
                sigma2 = sigma2 * (1 + epsilon * (np.exp(log_alpha_mala) - alpha_star))

        # Initialize R, sigma_R2, and compute log_pi and grad_log_pi for theta_curr
        R = self.covariance
        sigma_R2 = sigma2
        log_target_current = self.log_target_pdf(current_sample)
        grad_log_target_current = self.grad_log_target_pdf(current_sample)

        # Main loop
        for i in trange(self.nits, disable=not self.verbose):
            # Propose sample
            eta = np.random.normal(size=self.sample_dim)
            proposed_sample = (
                current_sample
                + (sigma_R2 / 2) * np.dot(R, np.dot(R.T, grad_log_target_current))
                + np.sqrt(sigma_R2) * np.dot(R, eta)
            )
            log_target_proposed = self.log_target_pdf(proposed_sample)
            grad_log_target_proposed = self.grad_log_target_pdf(proposed_sample)

            # Compute acceptance probability
            h_current_proposed = self.proposal_log_correction(
                current_sample, proposed_sample, sigma_R2, grad_log_target_proposed
            )
            h_proposed_current = self.proposal_log_correction(
                proposed_sample, current_sample, sigma_R2, grad_log_target_current
            )
            log_ratio = (
                log_target_proposed
                + h_current_proposed
                - log_target_current
                - h_proposed_current
            )
            log_alpha = np.minimum(0, log_ratio)

            # Adapt R and sigma2
            s_n_delta = np.sqrt(np.exp(log_alpha)) * (
                grad_log_target_proposed - grad_log_target_current
            )
            if i == 0:
                R = self.fisher_precondition_update(s_n_delta, R / np.sqrt(lambda_))
            else:
                R = self.fisher_precondition_update(s_n_delta, R)

            sigma2 = sigma2 * (1 + epsilon * (np.exp(log_alpha) - alpha_star))
            sigma_R2 = sigma2 / (np.trace(np.dot(R, R.T)) / self.sample_dim)

            # Accept/Reject
            if np.log(np.random.rand()) < log_alpha:
                current_sample = proposed_sample
                log_target_current = log_target_proposed
                grad_log_target_current = grad_log_target_proposed
                nacc += 1

            self.store[i] = current_sample

        self.acc = nacc / self.nits


class SimulatedAnnealingMetropolisHastings(MCMCAlgorithmBase):
    def sample(
        self,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.999,
        epsilon: float = 1.0,
    ) -> None:
        """
        Simulated Annealing MH algorithm with Gaussian proposal distribution.

        Args:
            initial_temp (float, optional): Initial temperature. Defaults to 10.0.
            cooling_rate (float, optional): Cooling rate. Defaults to 0.999.
            epsilon (float, optional): Standard deviation of the proposal distribution. Defaults to 1.0.
        """
        nacc = 0
        current_sample = self.init_sample
        log_target_current = self.log_target_pdf(current_sample)
        self.store[0, :] = current_sample
        current_temp = initial_temp

        for i in trange(self.nits, disable=not self.verbose):
            proposed_sample = current_sample + epsilon * np.random.normal(
                size=self.sample_dim
            )
            log_target_proposed = self.log_target_pdf(proposed_sample)
            log_ratio = (log_target_proposed - log_target_current) / current_temp
            log_alpha = np.minimum(0, log_ratio)

            if np.log(np.random.uniform()) < log_alpha:
                current_sample = proposed_sample
                log_target_current = log_target_proposed
                nacc += 1

            self.store[i + 1, :] = current_sample
            current_temp *= cooling_rate

        self.acc = nacc / self.nits


class HamiltonianMonteCarlo(MCMCAlgorithmBase):
    def __init__(
        self,
        log_target_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        grad_log_target_pdf: Callable[
            [npt.NDArray[np.floating]], npt.NDArray[np.floating]
        ],
        init_sample: npt.NDArray[np.floating],
        covariance: Optional[npt.NDArray[np.floating]] = None,
        nits: int = 5_000,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MCMC object. The object can be used to run various MCMC algorithms.

        Args:
            log_target_pdf (Callable[[T], T]): Logarithm of the target distribution.
            grad_log_target_pdf (Callable[[T], T]): Gradient of the logarithm of the target distribution.
            init_sample (npt.NDArray[np.floating]): Starting point of the chain.
            covariance (Optional[npt.NDArray[np.floating]], optional): Covariance matrix of the proposal distribution. Defaults to None.
            nits (int, optional): Number of iterations. Defaults to 5000.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
        """
        super().__init__(log_target_pdf, init_sample, covariance, nits, verbose)

        self.grad_log_target_pdf = grad_log_target_pdf

    def sample(self, delta: float = 0.14, L: int = 20) -> None:
        """
        Hamiltonian Monte Carlo Algorithm with leapfrog integrator.

        Args:
            delta (float, optional): Step size. Defaults to 0.14.
            L (int, optional): Number of leapfrog steps. Defaults to 20.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
        """
        self.store[0, :] = self.init_sample
        nacc = 0

        for t in trange(1, self.nits, disable=not self.verbose):
            # Current sample
            current_sample = self.store[t - 1, :]

            # Initialize momentum
            initial_momentum = np.random.randn(self.sample_dim)
            momentum = initial_momentum + delta / 2 * self.grad_log_target_pdf(
                current_sample
            )

            # Initialize proposed sample
            proposed_sample = current_sample + delta * momentum

            # Leapfrog updates
            for _ in range(L - 1):
                momentum = momentum + delta / 2 * self.grad_log_target_pdf(
                    proposed_sample
                )  # Half-step update for momentum
                proposed_sample = (
                    proposed_sample + delta * momentum
                )  # Full-step update for sample
                momentum = momentum + delta / 2 * self.grad_log_target_pdf(
                    proposed_sample
                )  # Half-step update for momentum

            # Reverse momentum for symmetry
            momentum = -momentum
            momentum = momentum + delta / 2 * self.grad_log_target_pdf(
                proposed_sample
            )  # Final half-step update for momentum

            # Compute Hamiltonian components
            initial_potential_energy = -self.log_target_pdf(current_sample)
            proposed_potential_energy = -self.log_target_pdf(proposed_sample)
            initial_kinetic_energy = -self.log_proposal_pdf(
                initial_momentum, np.zeros(self.sample_dim), np.eye(self.sample_dim)
            )
            proposed_kinetic_energy = -self.log_proposal_pdf(
                momentum, np.zeros(self.sample_dim), np.eye(self.sample_dim)
            )

            # Metropolis acceptance criterion
            log_ratio = (
                initial_potential_energy
                + initial_kinetic_energy
                - proposed_potential_energy
                - proposed_kinetic_energy
            )
            log_alpha = np.minimum(0, log_ratio)

            if np.log(np.random.uniform(0, 1)) < log_alpha:
                self.store[t, :] = proposed_sample
                nacc += 1
            else:
                self.store[t, :] = current_sample

        self.acc = nacc / (self.nits - 1)


class MCMCFactory:
    """
    Factory class to create MCMC algorithm instances.
    """

    _algorithm_map: Dict[str, Type[MCMCAlgorithmBase]] = {
        "rwm": RandomWalkMetropolisHastings,
        "am": AdaptiveMetropolisHastings,
        "mala": MetropolisAdjustedLangevinAlgorithm,
        "tmala": TamedMetropolisAdjustedLangevinAlgorithm,
        "tmalac": TamedMetropolisAdjustedLangevinAlgorithmCoordinatewise,
        "famala": FisherAdaptiveLangevinMetropolisHastings,
        "samh": SimulatedAnnealingMetropolisHastings,
        "hmc": HamiltonianMonteCarlo,
    }

    @staticmethod
    def create(
        algorithm_name: str,
        log_target_pdf: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        init_sample: npt.NDArray[np.floating],
        grad_log_target_pdf: Optional[
            Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
        ] = None,
        covariance: Optional[npt.NDArray[np.floating]] = None,
        nits: int = 5_000,
        verbose: bool = True,
    ) -> MCMCAlgorithmBase:
        """
        Create an instance of the specified MCMC algorithm.

        Args:
            algorithm_name (str): The name of the algorithm.
            log_target_pdf (Callable[[np.ndarray], float]): Logarithm of the target distribution.
            init_sample (np.ndarray): Starting point of the chain.
            covariance (Optional[np.ndarray]): Covariance matrix of the proposal distribution.
            nits (int, optional): Number of iterations. Defaults to 5000.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.

        Returns:
            MCMCAlgorithmBase: An instance of the specified MCMC algorithm.

        Raises:
            ValueError: If the algorithm name is not recognized.
        """
        algorithm_name_lower = algorithm_name.lower()

        # Get the algorithm class
        algorithm_class = MCMCFactory._algorithm_map.get(algorithm_name_lower)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        # Create the algorithm instance
        if "mala" in algorithm_name_lower or "hmc" in algorithm_name_lower:
            return algorithm_class(
                log_target_pdf=log_target_pdf,
                grad_log_target_pdf=grad_log_target_pdf,
                init_sample=init_sample,
                covariance=covariance,
                nits=nits,
                verbose=verbose,
            )
        else:
            return algorithm_class(
                log_target_pdf=log_target_pdf,
                init_sample=init_sample,
                covariance=covariance,
                nits=nits,
                verbose=verbose,
            )
