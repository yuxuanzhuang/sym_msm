import numpy as np

import torch

from typing import Optional, List

from tqdm.notebook import tqdm  # progress bar
from ..deepmsm.deepmsm import *

from typing import Optional, Union, Callable, Tuple
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from deeptime.decomposition.deep import vamp_score, vampnet_loss


def vamp_score_rev(
    data: torch.Tensor,
    data_lagged: torch.Tensor,
    method="VAMP2",
    epsilon: float = 1e-6,
    mode="trunc",
):
    r"""Computes the VAMP score based on data and corresponding time-shifted data.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert (
        method in valid_score_methods
    ), f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, (
        f"Data and data_lagged must be of same shape but were {data.shape} "
        f"and {data_lagged.shape}."
    )
    out = None
    if method == "VAMP1":
        koopman = koopman_matrix_rev(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.norm(koopman, p="nuc")
    elif method == "VAMP2":
        koopman = koopman_matrix_rev(data, data_lagged, epsilon=epsilon, mode=mode)
        out = torch.pow(torch.norm(koopman, p="fro"), 2)
    elif method == "VAMPE":
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        c0t = 0.5 * (c0t + c0t.t())
        c00 = 0.5 * (c00 + ctt.t())
        ctt = c00

        c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)

        out = torch.trace(
            2.0 * multi_dot([s, u_t, c0t, v])
            - multi_dot([s, u_t, c00, u, s, v_t, ctt, v])
        )
    assert out is not None
    return 1 + out


def koopman_matrix_rev(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1e-6,
    mode: str = "trunc",
    c_xx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Computes the reversible Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        c00, c0t, ctt = covariances(x, y, remove_mean=True)
    c0t = 0.5 * (c0t + c0t.t())
    c00 = 0.5 * (c00 + ctt.t())
    ctt = c00

    c00_sqrt_inv = sym_inverse(c00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(ctt, return_sqrt=True, epsilon=epsilon, mode=mode)
    return multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()


def vamp_score_sym(
    data: torch.Tensor,
    data_lagged: torch.Tensor,
    symmetry_fold: int,
    reversible: bool = False,
    method="VAMP2",
    epsilon: float = 1e-6,
    mode="trunc",
):
    r"""Computes the VAMP score based on data and corresponding time-shifted data with symmetry.

    Parameters
    ----------
    data : torch.Tensor
        (N, d)-dimensional torch tensor
    data_lagged : torch.Tensor
        (N, k)-dimensional torch tensor
    symmetry_fold : int
        fold of symmetry
    method : str, default='VAMP2'
        The scoring method. See :meth:`score <deeptime.decomposition.CovarianceKoopmanModel.score>` for details.
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues, alternatively regularization parameter.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.

    Returns
    -------
    score : torch.Tensor
        The score. It contains a contribution of :math:`+1` for the constant singular function since the
        internally estimated Koopman operator is defined on a decorrelated basis set.
    """
    assert (
        method in valid_score_methods
    ), f"Invalid method '{method}', supported are {valid_score_methods}"
    assert data.shape == data_lagged.shape, (
        f"Data and data_lagged must be of same shape but were {data.shape} "
        f"and {data_lagged.shape}."
    )
    out = None
    if method == "VAMP1":
        koopman = koopman_matrix_sym(
            data,
            data_lagged,
            symmetry_fold,
            reversible=reversible,
            epsilon=epsilon,
            mode=mode,
        )
        out = torch.norm(koopman, p="nuc")
    elif method == "VAMP2":
        koopman = koopman_matrix_sym(
            data,
            data_lagged,
            symmetry_fold,
            reversible=reversible,
            epsilon=epsilon,
            mode=mode,
        )
        out = torch.pow(torch.norm(koopman, p="fro"), 2)
    elif method == "VAMPE":
        c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
        if reversible:
            c0t = 0.5 * (c0t + c0t.t())
            c00 = 0.5 * (c00 + ctt.t())
            ctt = c00

        if c00.shape[0] % symmetry_fold != 0:
            raise ValueError(
                f"Number of features {c00.shape[0]} must"
                + f"be divisible by symmetry_fold {symmetry_fold}."
            )
        subset_rank = c00.shape[0] // symmetry_fold

        cov_00_blocks = []
        cov_0t_blocks = []
        cov_tt_blocks = []
        for i in range(symmetry_fold):
            cov_00_blocks.append(
                c00[:subset_rank, i * subset_rank : (i + 1) * subset_rank]
            )
            cov_0t_blocks.append(
                c0t[:subset_rank, i * subset_rank : (i + 1) * subset_rank]
            )
            cov_tt_blocks.append(
                ctt[:subset_rank, i * subset_rank : (i + 1) * subset_rank]
            )

        #        cov_00 = covariances.cov_00[:subset_rank, :subset_rank]
        #        cov_0t = covariances.cov_0t[:subset_rank, :subset_rank]
        #        cov_tt = covariances.cov_tt[:subset_rank, :subset_rank]
        cov_00 = torch.sum(torch.stack(cov_00_blocks), axis=0)
        cov_0t = torch.sum(torch.stack(cov_0t_blocks), axis=0)
        cov_tt = torch.sum(torch.stack(cov_tt_blocks), axis=0)

        c00_sqrt_inv = sym_inverse(cov_00, epsilon=epsilon, return_sqrt=True, mode=mode)
        ctt_sqrt_inv = sym_inverse(cov_tt, epsilon=epsilon, return_sqrt=True, mode=mode)
        koopman = multi_dot([c00_sqrt_inv, cov_0t, ctt_sqrt_inv]).t()

        u, s, v = torch.svd(koopman)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(ctt_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)

        out = torch.trace(
            2.0 * multi_dot([s, u_t, cov_0t, v])
            - multi_dot([s, u_t, cov_00, u, s, v_t, cov_tt, v])
        )
    assert out is not None
    return 1 + out


def koopman_matrix_sym(
    x: torch.Tensor,
    y: torch.Tensor,
    symmetry_fold: int,
    reversible: bool = False,
    epsilon: float = 1e-6,
    mode: str = "trunc",
    c_xx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Computes the Koopman matrix

    .. math:: K = C_{00}^{-1/2}C_{0t}C_{tt}^{-1/2}

    based on data over which the covariance matrices :math:`C_{\cdot\cdot}` are computed.

    Parameters
    ----------
    x : torch.Tensor
        Instantaneous data.
    y : torch.Tensor
        Time-lagged data.
    symmetry_fold : int
        fold of symmetry
    epsilon : float, default=1e-6
        Cutoff parameter for small eigenvalues.
    mode : str, default='trunc'
        Regularization mode for Hermetian inverse. See :meth:`sym_inverse`.
    c_xx : tuple of torch.Tensor, optional, default=None
        Tuple containing c00, c0t, ctt if already computed.

    Returns
    -------
    K : torch.Tensor
        The Koopman matrix.
    """
    if c_xx is not None:
        c00, c0t, ctt = c_xx
    else:
        c00, c0t, ctt = covariances(x, y, remove_mean=True)
    if reversible:
        c0t = 0.5 * (c0t + c0t.t())
        c00 = 0.5 * (c00 + ctt.t())
        ctt = c00

    if c00.shape[0] % symmetry_fold != 0:
        raise ValueError(
            f"Number of features {c00.shape[0]} must"
            + f"be divisible by symmetry_fold {symmetry_fold}."
        )
    subset_rank = c00.shape[0] // symmetry_fold

    cov_00_blocks = []
    cov_0t_blocks = []
    cov_tt_blocks = []
    for i in range(symmetry_fold):
        cov_00_blocks.append(c00[:subset_rank, i * subset_rank : (i + 1) * subset_rank])
        cov_0t_blocks.append(c0t[:subset_rank, i * subset_rank : (i + 1) * subset_rank])
        cov_tt_blocks.append(ctt[:subset_rank, i * subset_rank : (i + 1) * subset_rank])

    #        cov_00 = covariances.cov_00[:subset_rank, :subset_rank]
    #        cov_0t = covariances.cov_0t[:subset_rank, :subset_rank]
    #        cov_tt = covariances.cov_tt[:subset_rank, :subset_rank]
    cov_00 = torch.sum(torch.stack(cov_00_blocks), axis=0)
    cov_0t = torch.sum(torch.stack(cov_0t_blocks), axis=0)
    cov_tt = torch.sum(torch.stack(cov_tt_blocks), axis=0)

    c00_sqrt_inv = sym_inverse(cov_00, return_sqrt=True, epsilon=epsilon, mode=mode)
    ctt_sqrt_inv = sym_inverse(cov_tt, return_sqrt=True, epsilon=epsilon, mode=mode)
    return multi_dot([c00_sqrt_inv, cov_0t, ctt_sqrt_inv]).t()
