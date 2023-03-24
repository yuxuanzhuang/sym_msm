"""
@author: yuxuanzhuang
"""

from collections import namedtuple
from numbers import Integral
from typing import Optional, Union, Callable

import numpy as np

from deeptime.decomposition import (
    VAMP,
    TICA,
    CovarianceKoopmanModel,
    TransferOperatorModel,
)
from deeptime.base import EstimatorTransformer
from deeptime.basis import Identity, Observable, Concatenation
from deeptime.covariance import Covariance, CovarianceModel
from deeptime.numeric import spd_inv_split, eig_corr
from deeptime.util.types import (
    to_dataset,
    is_timelagged_dataset,
    ensure_timeseries_data,
)
from deeptime.util.data import TimeLaggedDataset, TrajectoryDataset, TrajectoriesDataset
from typing import Tuple, Optional, Union, List


class SymVAMP(VAMP):
    r"""Variational approach for Markov processes (VAMP) with a symmetric observable transform."""

    def __init__(
        self,
        symmetry_fold: int = 1,
        lagtime: Optional[int] = None,
        dim: Optional[int] = None,
        var_cutoff: Optional[float] = None,
        scaling: Optional[str] = None,
        epsilon: float = 1e-6,
        observable_transform: Callable[[np.ndarray], np.ndarray] = Identity(),
    ):
        super().__init__(
            lagtime=lagtime,
            dim=dim,
            var_cutoff=var_cutoff,
            scaling=scaling,
            epsilon=epsilon,
            observable_transform=observable_transform,
        )
        self.symmetry_fold = symmetry_fold

    _DiagonalizationResults = namedtuple(
        "DiagonalizationResults",
        [
            "rank0",
            "rankt",
            "singular_values",
            "left_singular_vecs",
            "right_singular_vecs",
            "left_singular_vecs_full",
            "right_singular_vecs_full",
            "cov",
        ],
    )

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim, var_cutoff, symmetry_fold) -> _DiagonalizationResults:
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model."""
        if covariances.cov_00.shape[0] % symmetry_fold != 0:
            raise ValueError(
                f"Number of features {covariances.cov_00.shape[0]} must"
                + f"be divisible by symmetry_fold {symmetry_fold}."
            )

        subset_rank = covariances.cov_00.shape[0] // symmetry_fold
        cov_00_blocks = []
        cov_0t_blocks = []
        cov_tt_blocks = []
        for i in range(symmetry_fold):
            cov_00_blocks.append(covariances.cov_00[:subset_rank, i * subset_rank : (i + 1) * subset_rank])
            cov_0t_blocks.append(covariances.cov_0t[:subset_rank, i * subset_rank : (i + 1) * subset_rank])
            cov_tt_blocks.append(covariances.cov_tt[:subset_rank, i * subset_rank : (i + 1) * subset_rank])

        #        cov_00 = covariances.cov_00[:subset_rank, :subset_rank]
        #        cov_0t = covariances.cov_0t[:subset_rank, :subset_rank]
        #        cov_tt = covariances.cov_tt[:subset_rank, :subset_rank]
        cov_00 = np.sum(cov_00_blocks, axis=0)
        cov_0t = np.sum(cov_0t_blocks, axis=0)
        cov_tt = np.sum(cov_tt_blocks, axis=0)

        cov = CovarianceModel(
            cov_00=cov_00,
            cov_0t=cov_0t,
            cov_tt=cov_tt,
            mean_0=covariances.mean_0[:subset_rank],
            mean_t=covariances.mean_t[:subset_rank],
        )

        L0 = spd_inv_split(cov_00, epsilon=epsilon)
        rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(cov_tt, epsilon=epsilon)
        rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, cov_0t).dot(Lt)
        from scipy.linalg import svd

        A, s, BT = svd(W, compute_uv=True, lapack_driver="gesvd")

        singular_values = s

        m = CovarianceKoopmanModel.effective_output_dimension(rank0, rankt, dim, var_cutoff, singular_values)

        U = np.dot(L0, A[:, :m])
        V = np.dot(Lt, BT[:m, :].T)

        # scale vectors
        if scaling is not None and scaling in ("km", "kinetic_map"):
            U *= s[np.newaxis, 0:m]  # scaled left singular functions induce a kinetic map
            V *= s[
                np.newaxis, 0:m
            ]  # scaled right singular functions induce a kinetic map wrt. backward propagator

        return SymVAMP._DiagonalizationResults(
            rank0=rank0,
            rankt=rankt,
            singular_values=singular_values,
            left_singular_vecs_full=np.tile(U, symmetry_fold),
            right_singular_vecs_full=np.tile(V, symmetry_fold),
            left_singular_vecs=U,
            right_singular_vecs=V,
            cov=cov,
        )

    def _decompose(self, covariances: CovarianceModel):
        decomposition = self._decomposition(
            covariances,
            self.epsilon,
            self.scaling,
            self.dim,
            self.var_cutoff,
            self.symmetry_fold,
        )
        return SymCovarianceKoopmanModel(
            symmetrty_fold=self.symmetry_fold,
            instantaneous_coefficients=decomposition.left_singular_vecs,
            singular_values=decomposition.singular_values,
            timelagged_coefficients=decomposition.right_singular_vecs,
            instantaneous_coefficients_full=decomposition.left_singular_vecs_full,
            timelagged_coefficients_full=decomposition.right_singular_vecs_full,
            rank_0=decomposition.rank0,
            rank_t=decomposition.rankt,
            dim=self.dim,
            var_cutoff=self.var_cutoff,
            cov=decomposition.cov,
            cov_full=covariances,
            scaling=self.scaling,
            epsilon=self.epsilon,
            instantaneous_obs=self.observable_transform,
            timelagged_obs=self.observable_transform,
        )

    def fetch_model(self) -> "SymCovarianceKoopmanModel":
        r"""Finalizes current model and yields new :class:`SymCovarianceKoopmanModel`.
        Returns
        -------
        model : SymCovarianceKoopmanModel
            The estimated model.
        """
        if self._covariance_estimator is not None:
            # This can only occur when partial_fit was called.
            # A call to fit, fit_from_timeseries, fit_from_covariances ultimately always leads to a call to
            # fit_from_covariances which sets the self._covariance_estimator to None.
            self._model = self._decompose(self._covariance_estimator.fetch_model())
            self._covariance_estimator = None
        return self._model

    def transform_subunit(self, data, propagate=False):
        r"""Projects given timeseries onto dominant singular functions. This method dispatches to
        :meth:`CovarianceKoopmanModel.transform_subunit`.

        Parameters
        ----------
        data : (T, n) ndarray
            Input timeseries data.
        propagate : bool, default=False
            Whether to apply the Koopman operator after data was transformed into the whitened feature space.

        Returns
        -------
        Y : (T, m) ndarray
            The projected data.
            If `right` is True, projection will be on the right singular functions. Otherwise, projection will be on
            the left singular functions.
        """
        return self.fetch_model().transform_subunit(data, propagate=propagate)


class SymTICA(TICA, SymVAMP):
    def __init__(
        self,
        symmetry_fold,
        lagtime: Optional[int] = None,
        epsilon: float = 1e-6,
        dim: Optional[int] = None,
        var_cutoff: Optional[float] = None,
        scaling: Optional[str] = "kinetic_map",
        observable_transform: Callable[[np.ndarray], np.ndarray] = Identity(),
    ):
        SymVAMP.__init__(
            self,
            symmetry_fold=symmetry_fold,
            lagtime=lagtime,
            dim=dim,
            var_cutoff=var_cutoff,
            scaling=scaling,
            epsilon=epsilon,
            observable_transform=observable_transform,
        )

    @staticmethod
    def _decomposition(
        covariances, epsilon, scaling, dim, var_cutoff, symmetry_fold
    ) -> SymVAMP._DiagonalizationResults:
        print("symmetry_fold", symmetry_fold)
        if covariances.cov_00.shape[0] % symmetry_fold != 0:
            raise ValueError(
                f"Number of features {covariances.cov_00.shape[0]} must"
                + f"be divisible by symmetry_fold {symmetry_fold}."
            )

        subset_rank = covariances.cov_00.shape[0] // symmetry_fold
        # convert covariance matrices to blocks
        cov_00_blocks = []
        cov_0t_blocks = []
        cov_tt_blocks = []
        for i in range(symmetry_fold):
            cov_00_blocks.append(covariances.cov_00[:subset_rank, i * subset_rank : (i + 1) * subset_rank])
            cov_0t_blocks.append(covariances.cov_0t[:subset_rank, i * subset_rank : (i + 1) * subset_rank])
            cov_tt_blocks.append(covariances.cov_tt[:subset_rank, i * subset_rank : (i + 1) * subset_rank])

        #        cov_00 = covariances.cov_00[:subset_rank, :subset_rank]
        #        cov_0t = covariances.cov_0t[:subset_rank, :subset_rank]
        #        cov_tt = covariances.cov_tt[:subset_rank, :subset_rank]
        cov_00 = np.sum(cov_00_blocks, axis=0)
        cov_0t = np.sum(cov_0t_blocks, axis=0)
        cov_tt = np.sum(cov_tt_blocks, axis=0)

        cov = CovarianceModel(
            cov_00=cov_00,
            cov_0t=cov_0t,
            cov_tt=cov_tt,
            mean_0=covariances.mean_0[:subset_rank],
            mean_t=covariances.mean_t[:subset_rank],
            bessels_correction=covariances.bessels_correction,
            symmetrized=covariances.symmetrized,
            lagtime=covariances.lagtime,
            data_mean_removed=covariances.data_mean_removed,
        )

        from deeptime.numeric import ZeroRankError

        # diagonalize with low rank approximation
        try:
            eigenvalues, eigenvectors, rank = eig_corr(
                cov_00, cov_0t, epsilon, canonical_signs=True, return_rank=True
            )
        except ZeroRankError:
            raise ZeroRankError(
                "All input features are constant in all time steps. "
                "No dimension would be left after dimension reduction."
            )
        if scaling in ("km", "kinetic_map"):  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        elif scaling == "commute_map":  # scale by (regularized) timescales
            lagtime = covariances.lagtime
            timescales = 1.0 - lagtime / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = (
                0.5 * timescales * np.maximum(np.tanh(np.pi * ((timescales - lagtime) / lagtime) + 1), 0)
            )

            eigenvectors *= np.sqrt(regularized_timescales / 2)

        return SymVAMP._DiagonalizationResults(
            rank0=rank,
            rankt=rank,
            singular_values=eigenvalues,
            left_singular_vecs_full=np.tile(eigenvectors, symmetry_fold),
            right_singular_vecs_full=np.tile(eigenvectors, symmetry_fold),
            left_singular_vecs=eigenvectors,
            right_singular_vecs=eigenvectors,
            cov=cov,
        )


class SymWhiteningTransform(Observable):
    r"""Transformation of symmetric data into a whitened space.
    It is assumed that for a covariance matrix :math:`C` the
    square-root inverse :math:`C^{-1/2}` was already computed. Optionally a mean :math:`\mu` can be provided.
    This yields the transformation
    .. math::
        y = C^{-1/2}(x-\mu).
    Parameters
    ----------
    sqrt_inv_cov : (n, k) ndarray
        Square-root inverse of covariance matrix.
    mean : (n, ) ndarray, optional, default=None
        The mean if it should be subtracted.
    dim : int, optional, default=None
        Additional restriction in the dimension, removes all but the first `dim` components of the output.
    See Also
    --------
    deeptime.numeric.spd_inv_sqrt : Method to obtain (regularized) inverses of covariance matrices.
    """

    def __init__(
        self,
        sqrt_inv_cov: np.ndarray,
        mean: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
    ):
        self.sqrt_inv_cov = sqrt_inv_cov
        self.mean = mean
        self.dim = dim

    def _evaluate(self, x: np.ndarray, nosum=False):
        if self.mean is not None:
            x = x - self.mean
        if nosum:
            return x @ self.sqrt_inv_cov[..., : self.dim]
        return np.sum(x @ self.sqrt_inv_cov[..., : self.dim], axis=1)


class SymWhiteningTransform_Multimer(Observable):
    r"""Transformation of symmetric data into a whitened space.
    It is assumed that for a covariance matrix :math:`C` the
    square-root inverse :math:`C^{-1/2}` was already computed. Optionally a mean :math:`\mu` can be provided.
    This yields the transformation
    .. math::
        y = C^{-1/2}(x-\mu).
    Parameters
    ----------
    sqrt_inv_cov : (n, k) ndarray
        Square-root inverse of covariance matrix.
    mean : (n, ) ndarray, optional, default=None
        The mean if it should be subtracted.
    dim : int, optional, default=None
        Additional restriction in the dimension, removes all but the first `dim` components of the output.
    See Also
    --------
    deeptime.numeric.spd_inv_sqrt : Method to obtain (regularized) inverses of covariance matrices.
    """

    def __init__(
        self,
        sqrt_inv_cov: np.ndarray,
        mean: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
    ):
        self.sqrt_inv_cov = sqrt_inv_cov
        self.mean = mean
        self.dim = dim

    def _evaluate(self, x: np.ndarray):
        if self.mean is not None:
            x = x - self.mean
        return x @ self.sqrt_inv_cov[..., : self.dim]


class SymCovarianceKoopmanModel(CovarianceKoopmanModel, TransferOperatorModel):
    """
    Symmetric transformation version of Covariance Koopman Model
    """

    def __init__(
        self,
        symmetrty_fold,
        instantaneous_coefficients,
        singular_values,
        timelagged_coefficients,
        instantaneous_coefficients_full,
        timelagged_coefficients_full,
        cov,
        cov_full,
        rank_0: int,
        rank_t: int,
        dim=None,
        var_cutoff=None,
        scaling=None,
        epsilon=1e-10,
        instantaneous_obs: Callable[[np.ndarray], np.ndarray] = Identity(),
        timelagged_obs: Callable[[np.ndarray], np.ndarray] = Identity(),
    ):
        self.symmetry_fold = symmetrty_fold
        self._whitening_instantaneous = SymWhiteningTransform(
            instantaneous_coefficients,
            cov.mean_0[: instantaneous_coefficients.shape[0]] if cov.data_mean_removed else None,
        )
        self._whitening_timelagged = SymWhiteningTransform(
            timelagged_coefficients,
            cov.mean_t[: timelagged_coefficients.shape[0]] if cov.data_mean_removed else None,
        )

        self._whitening_instantaneous_full = SymWhiteningTransform(
            instantaneous_coefficients_full,
            cov_full.mean_0[: instantaneous_coefficients.shape[0]] if cov.data_mean_removed else None,
        )
        self._whitening_timelagged_full = SymWhiteningTransform(
            timelagged_coefficients_full,
            cov_full.mean_t[: timelagged_coefficients_full.shape[0]] if cov.data_mean_removed else None,
        )
        TransferOperatorModel.__init__(
            self,
            np.diag(singular_values),
            Concatenation_Multimer(self._whitening_instantaneous, instantaneous_obs, self.symmetry_fold),
            Concatenation_Multimer(self._whitening_timelagged, timelagged_obs, self.symmetry_fold),
        )
        self._whitening_instantaneous_sub = SymWhiteningTransform_Multimer(
            instantaneous_coefficients,
            cov.mean_0[: instantaneous_coefficients.shape[0]] if cov.data_mean_removed else None,
        )
        self._whitening_timelagged_sub = SymWhiteningTransform_Multimer(
            timelagged_coefficients,
            cov.mean_t[: timelagged_coefficients.shape[0]] if cov.data_mean_removed else None,
        )

        self.instantaneous_obs_sub = Concatenation_Multimer(
            self._whitening_instantaneous_sub, instantaneous_obs, self.symmetry_fold
        )
        self.timelagged_obs_sub = Concatenation_Multimer(
            self._whitening_timelagged_sub, timelagged_obs, self.symmetry_fold
        )

        self._instantaneous_coefficients = instantaneous_coefficients
        self._timelagged_coefficients = timelagged_coefficients
        self._instantaneous_coefficients_full = instantaneous_coefficients_full
        self._timelagged_coefficients_full = timelagged_coefficients_full
        self._singular_values = singular_values
        self._cov = cov
        self._cov_full = cov_full

        self._scaling = scaling
        self._epsilon = epsilon
        self._rank_0 = rank_0
        self._rank_t = rank_t
        self._dim = dim
        self._var_cutoff = var_cutoff
        self._update_output_dimension()

    def transform(self, data: np.ndarray, **kw):
        #        data = data.reshape(data.shape[0], self.symmetry_fold, -1)
        return self.instantaneous_obs(data)

    def transform_subunit(self, data: np.ndarray, **kw):
        #        data = data.reshape(data.shape[0], self.symmetry_fold, -1)
        return self.instantaneous_obs_sub(data).reshape(data.shape[0], self.symmetry_fold, -1)


class Concatenation_Multimer(Concatenation):
    r"""Concatenation operation to evaluate :math:`(f_1 \circ f_2)(x) = f_1(f_2(x))`, where
    :math:`f_1` and :math:`f_2` are observables.

    Parameters
    ----------
    obs1 : Callable
        First observable :math:`f_1`.
    obs2 : Callable
        Second observable :math:`f_2`.
    """

    def __init__(
        self,
        obs1: Callable[[np.ndarray], np.ndarray],
        obs2: Callable[[np.ndarray], np.ndarray],
        symmetry_fold: int,
    ):
        self.obs1 = obs1
        self.obs2 = obs2
        self.symmetry_fold = symmetry_fold

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        result = self.obs2(x)
        result = result.reshape(result.shape[0], self.symmetry_fold, -1)
        return self.obs1(result)


# TODO: rewrite covaraince class to be able to handle multimer data
class SymVAMP_NOAUG(SymVAMP):
    r"""Variational approach for Markov processes (VAMP) with a symmetric observable transform."""

    def __init__(
        self,
        symmetry_fold: int = 1,
        lagtime: Optional[int] = None,
        dim: Optional[int] = None,
        var_cutoff: Optional[float] = None,
        scaling: Optional[str] = None,
        epsilon: float = 1e-6,
        observable_transform: Callable[[np.ndarray], np.ndarray] = Identity(),
    ):
        raise NotImplementedError("SymVAMP_NOAUG is not implemented yet.")
        super().__init__(
            lagtime=lagtime,
            dim=dim,
            var_cutoff=var_cutoff,
            scaling=scaling,
            epsilon=epsilon,
            observable_transform=observable_transform,
        )
        self.symmetry_fold = symmetry_fold

    _DiagonalizationResults = namedtuple(
        "DiagonalizationResults",
        [
            "rank0",
            "rankt",
            "singular_values",
            "left_singular_vecs",
            "right_singular_vecs",
            "left_singular_vecs_full",
            "right_singular_vecs_full",
            "cov",
        ],
    )

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim, var_cutoff, symmetry_fold) -> _DiagonalizationResults:
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model."""
        cov_00 = covariances.cov_00
        cov_0t = covariances.cov_0t
        cov_tt = covariances.cov_tt

        cov = CovarianceModel(
            cov_00=cov_00,
            cov_0t=cov_0t,
            cov_tt=cov_tt,
            mean_0=covariances.mean_0,
            mean_t=covariances.mean_t,
        )

        L0 = spd_inv_split(cov_00, epsilon=epsilon)
        rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(cov_tt, epsilon=epsilon)
        rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, cov_0t).dot(Lt)
        from scipy.linalg import svd

        A, s, BT = svd(W, compute_uv=True, lapack_driver="gesvd")

        singular_values = s

        m = CovarianceKoopmanModel.effective_output_dimension(rank0, rankt, dim, var_cutoff, singular_values)

        U = np.dot(L0, A[:, :m])
        V = np.dot(Lt, BT[:m, :].T)

        # scale vectors
        if scaling is not None and scaling in ("km", "kinetic_map"):
            U *= s[np.newaxis, 0:m]  # scaled left singular functions induce a kinetic map
            V *= s[
                np.newaxis, 0:m
            ]  # scaled right singular functions induce a kinetic map wrt. backward propagator

        return SymVAMP_NOAUG._DiagonalizationResults(
            rank0=rank0,
            rankt=rankt,
            singular_values=singular_values,
            left_singular_vecs_full=np.tile(U, symmetry_fold),
            right_singular_vecs_full=np.tile(V, symmetry_fold),
            left_singular_vecs=U,
            right_singular_vecs=V,
            cov=covariances,
        )

    def partial_fit(self, data):
        r"""Updates the covariance estimates through a new batch of data.

        Parameters
        ----------
        data : tuple(ndarray, ndarray)
            A tuple of ndarrays which have to have same shape and are :math:`X_t` and :math:`X_{t+\tau}`, respectively.
            Here, :math:`\tau` denotes the lagtime.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        if self._covariance_estimator is None:
            self._covariance_estimator = self.covariance_estimator(lagtime=self.lagtime)
        x, y = to_multimer_dataset(data, symmetry_fold=self.symmetry_fold, lagtime=self.lagtime)[:]
        self._covariance_estimator.partial_fit((self.observable_transform(x), self.observable_transform(y)))
        return self

    def fit_from_timeseries(self, data, weights=None):
        r"""Estimates a :class:`CovarianceKoopmanModel` directly from time-series data using the :class:`Covariance`
        estimator. For parameters `dim`, `scaling`, `epsilon`.

        Parameters
        ----------
        data
            Input data, see :meth:`to_dataset <deeptime.util.types.to_dataset>` for options.
        weights
            See the :class:`Covariance <deeptime.covariance.Covariance>` estimator.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        datasets = to_multimer_datasets(data, symmetry_fold=self.symmetry_fold, lagtime=self.lagtime)
        self._covariance_estimator = self.covariance_estimator(lagtime=self.lagtime)
        self.datasets = datasets
        for dataset in datasets:
            print(dataset)
            x = dataset.data
            y = dataset.data_lagged
            transformed = (self.observable_transform(x), self.observable_transform(y))
            self._covariance_estimator = self._covariance_estimator.partial_fit(transformed, weights=weights)
        covariances = self._covariance_estimator.fetch_model()
        return self.fit_from_covariances(covariances)

    def fit_from_covariances(self, covariances: Union[Covariance, CovarianceModel]):
        r"""Fits from existing covariance model (or covariance estimator containing model).

        Parameters
        ----------
        covariances : CovarianceModel or Covariance
            Covariance model containing covariances or Covariance estimator containing a covariance model. The model
            in particular has matrices :math:`C_{00}, C_{0t}, C_{tt}`.

        Returns
        -------
        self : VAMP
            Reference to self.
        """
        self._covariance_estimator = None
        covariances = self._to_covariance_model(covariances)
        self._model = self._decompose(covariances)
        return self

    def _decompose(self, covariances: CovarianceModel):
        decomposition = self._decomposition(
            covariances,
            self.epsilon,
            self.scaling,
            self.dim,
            self.var_cutoff,
            self.symmetry_fold,
        )

        return SymCovarianceKoopmanModel(
            symmetrty_fold=self.symmetry_fold,
            instantaneous_coefficients=decomposition.left_singular_vecs,
            singular_values=decomposition.singular_values,
            timelagged_coefficients=decomposition.right_singular_vecs,
            instantaneous_coefficients_full=decomposition.left_singular_vecs_full,
            timelagged_coefficients_full=decomposition.right_singular_vecs_full,
            rank_0=decomposition.rank0,
            rank_t=decomposition.rankt,
            dim=self.dim,
            var_cutoff=self.var_cutoff,
            cov=decomposition.cov,
            cov_full=covariances,
            scaling=self.scaling,
            epsilon=self.epsilon,
            instantaneous_obs=self.observable_transform,
            timelagged_obs=self.observable_transform,
        )


class SymTICA_NOAUG(SymVAMP_NOAUG, TICA):
    def __init__(
        self,
        symmetry_fold,
        lagtime: Optional[int] = None,
        epsilon: float = 1e-6,
        dim: Optional[int] = None,
        var_cutoff: Optional[float] = None,
        scaling: Optional[str] = "kinetic_map",
        observable_transform: Callable[[np.ndarray], np.ndarray] = Identity(),
    ):
        SymVAMP.__init__(
            self,
            symmetry_fold=symmetry_fold,
            lagtime=lagtime,
            dim=dim,
            var_cutoff=var_cutoff,
            scaling=scaling,
            epsilon=epsilon,
            observable_transform=observable_transform,
        )


def to_multimer_datasets(
    data: Union[TimeLaggedDataset, Tuple[np.ndarray, np.ndarray], np.ndarray],
    symmetry_fold: int,
    lagtime: Optional[int] = None,
):
    if isinstance(data, np.ndarray) and data.ndim >= 3:
        data_multimer = []
        # assume the third axis is the feature axis
        for x in np.split(data, symmetry_fold, axis=2):
            data_multimer.append(x)
            data = data_multimer

    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(f"If data is provided as tuple the length must be 2 but was {len(data)}.")

        datasets = []
        for x, y in zip(
            np.split(data[0], symmetry_fold, axis=1),
            np.split(data[1], symmetry_fold, axis=1),
        ):
            datasets.append(TimeLaggedDataset(x, y))
        return datasets

    if isinstance(data, np.ndarray):
        if lagtime is None:
            raise ValueError("In case data is a single trajectory the lagtime must be given.")
        return TrajectoriesDataset.from_numpy(
            lagtime, [data_sub for data_sub in np.split(data, symmetry_fold, axis=1)]
        )

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
        data = ensure_timeseries_data(data)
        traj_list = []
        for traj in data:
            traj_list.extend([traj_sub for traj_sub in np.split(traj, symmetry_fold, axis=1)])
        return TrajectoriesDataset.from_numpy(lagtime, traj_list)

    if isinstance(data, TrajectoryDataset):
        traj_list = []
        for data_sub in np.split(data.data, 2, axis=1):
            for data_lagged_sub in np.split(data.data_lagged, 2, axis=1):
                traj_list.append(TimeLaggedDataset(data_sub, data_lagged_sub))
        return traj_list
    assert hasattr(data, "__len__") and len(data) > 0, "Data is empty."

    assert is_timelagged_dataset(data), (
        "Data is not a time-lagged dataset, i.e., yielding tuples of instantaneous and time-lagged data. "
        "In case of multiple trajectories, deeptime.util.data.TrajectoriesDataset may be used."
    )

    return data
