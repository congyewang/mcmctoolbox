import numpy as np
from numpy import typing as npt

from mcmctoolbox.mcmc import (
    AdaptiveMetropolisHastings,
    FisherAdaptiveLangevinMetropolisHastings,
    HamiltonianMonteCarlo,
    MCMCFactory,
    MetropolisAdjustedLangevinAlgorithm,
    RandomWalkMetropolisHastings,
    SimulatedAnnealingMetropolisHastings,
    TamedMetropolisAdjustedLangevinAlgorithm,
    TamedMetropolisAdjustedLangevinAlgorithmCoordinatewise,
)


def mock_log_target_pdf(x: npt.NDArray[np.floating]) -> np.floating:
    return -0.5 * np.sum(x**2)


def mock_grad_log_target_pdf(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return -x


def test_random_walk_metropolis_hastings():
    initial_sample = np.array([0.0])
    rwmh = RandomWalkMetropolisHastings(mock_log_target_pdf, initial_sample)
    rwmh.sample(epsilon=0.1)
    assert rwmh.store.shape == (5001, 1)
    assert 0 <= rwmh.acc <= 1


def test_adaptive_metropolis_hastings():
    initial_sample = np.array([0.0])
    amh = AdaptiveMetropolisHastings(mock_log_target_pdf, initial_sample)
    amh.sample()
    assert amh.store.shape == (5001, 1)
    assert 0 <= amh.acc <= 1


def test_metropolis_adjusted_langevin_algorithm() -> None:
    initial_sample = np.array([0.0])
    mala = MetropolisAdjustedLangevinAlgorithm(
        mock_log_target_pdf, mock_grad_log_target_pdf, initial_sample
    )
    mala.sample(epsilon=0.1)
    assert mala.store.shape == (5001, 1)
    assert 0 <= mala.acc <= 1


def test_tamed_metropolis_adjusted_langevin_algorithm() -> None:
    initial_sample = np.array([0.0])
    tmala = TamedMetropolisAdjustedLangevinAlgorithm(
        mock_log_target_pdf, mock_grad_log_target_pdf, initial_sample
    )
    tmala.sample(epsilon=0.01)
    assert tmala.store.shape == (5001, 1)
    assert 0 <= tmala.acc <= 1


def test_tamed_metropolis_adjusted_langevin_algorithm_coordinatewise() -> None:
    initial_sample = np.array([0.0])
    tmalac = TamedMetropolisAdjustedLangevinAlgorithmCoordinatewise(
        mock_log_target_pdf, mock_grad_log_target_pdf, initial_sample
    )
    tmalac.sample(epsilon=0.01)
    assert tmalac.store.shape == (5001, 1)
    assert 0 <= tmalac.acc <= 1


def test_fisher_adaptive_langevin_metropolis_hastings() -> None:
    initial_sample = np.array([0.0])
    famala = FisherAdaptiveLangevinMetropolisHastings(
        mock_log_target_pdf, mock_grad_log_target_pdf, initial_sample
    )
    famala.sample()
    assert famala.store.shape == (5001, 1)
    assert 0 <= famala.acc <= 1


def test_simulated_annealing_metropolis_hastings() -> None:
    initial_sample = np.array([0.0])
    sammh = SimulatedAnnealingMetropolisHastings(mock_log_target_pdf, initial_sample)
    sammh.sample()
    assert sammh.store.shape == (5001, 1)
    assert 0 <= sammh.acc <= 1


def test_hamiltonian_monte_carlo() -> None:
    initial_sample = np.array([0.0])
    hmc = HamiltonianMonteCarlo(
        mock_log_target_pdf, mock_grad_log_target_pdf, initial_sample
    )
    hmc.sample()
    assert hmc.store.shape == (5001, 1)
    assert 0 <= hmc.acc <= 1


def test_mcmc_factory() -> None:
    initial_sample = np.array([0.0])
    rwmh = MCMCFactory.create("rwm", mock_log_target_pdf, initial_sample)
    assert isinstance(rwmh, RandomWalkMetropolisHastings)
    amh = MCMCFactory.create("am", mock_log_target_pdf, initial_sample)
    assert isinstance(amh, AdaptiveMetropolisHastings)
    mala = MCMCFactory.create(
        "mala", mock_log_target_pdf, initial_sample, mock_grad_log_target_pdf
    )
    assert isinstance(mala, MetropolisAdjustedLangevinAlgorithm)
    tmala = MCMCFactory.create(
        "tmala", mock_log_target_pdf, initial_sample, mock_grad_log_target_pdf
    )
    assert isinstance(tmala, TamedMetropolisAdjustedLangevinAlgorithm)
    tmalac = MCMCFactory.create(
        "tmalac", mock_log_target_pdf, initial_sample, mock_grad_log_target_pdf
    )
    assert isinstance(tmalac, TamedMetropolisAdjustedLangevinAlgorithmCoordinatewise)
    famala = MCMCFactory.create(
        "famala", mock_log_target_pdf, initial_sample, mock_grad_log_target_pdf
    )
    assert isinstance(famala, FisherAdaptiveLangevinMetropolisHastings)
    sammh = MCMCFactory.create("samh", mock_log_target_pdf, initial_sample)
    assert isinstance(sammh, SimulatedAnnealingMetropolisHastings)
