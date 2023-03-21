import warnings
from ..msm.MSM_a7 import MSMInitializer
from ..util.dataloader import MultimerTrajectoriesDataset

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from typing import Optional, List
from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset

from tqdm.notebook import tqdm  # progress bar
import deeptime
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.base import Model, Transformer
from deeptime.base_torch import DLEstimatorMixin
from deeptime.util.torch import map_data
from deeptime.markov.tools.analysis import pcca_memberships


from deeptime.decomposition.deep import VAMPNet
from ..deepmsm.deepmsm import *
from copy import deepcopy

from typing import Optional, Union, Callable, Tuple
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from sklearn import preprocessing

from .vampnet import VAMPNet_Multimer


class VAMPNet_Multimer_Rev_Model(VAMPNet_Multimer, Transformer):
    def __init__(
        self,
        multimer: int,
        n_states: int,
        lobe: nn.Module,
        lobe_timelagged: Optional[nn.Module] = None,
        ulayer: nn.Module = None,
        slayer: nn.Module = None,
        cg_list=None,
        mask=torch.nn.Identity(),
        device=None,
        optimizer: Union[str, Callable] = "Adam",
        learning_rate: float = 0.0005,
        score_method: str = "VAMP2",
        score_mode: str = "regularize",
        epsilon: float = 0.000001,
        dtype=np.float32,
    ):
        super().__init__(
            multimer,
            n_states,
            lobe,
            lobe_timelagged,
            device,
            optimizer,
            learning_rate,
            score_method,
            score_mode,
            epsilon,
            dtype,
        )
        self._ulayer = ulayer
        self._slayer = slayer
        if cg_list is not None:
            self._cg_list = cg_list
        self.mask = mask

    def transform(self, data, **kwargs):
        self._lobe.eval()
        net = self._lobe
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(net(self.mask(data_tensor)).cpu().numpy())
        return out if len(out) > 1 else out[0]

    def get_mu(self, data_t):
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            x_t = net(self.mask(torch.Tensor(data_t).to(self._device)))
            mu = self._ulayer(x_t, x_t, return_mu=True)[-1]  # use dummy x_0
        return mu.detach().to("cpu").numpy()

    def get_transition_matrix(self, data_0, data_t):
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            x_0 = net(self.mask(torch.Tensor(data_0).to(self._device)))
            x_t = net(self.mask(torch.Tensor(data_t).to(self._device)))
            _, K = vampe_loss_rev(x_0, x_t, self._ulayer, self._slayer, return_K=True)

        K = K.to("cpu").numpy().astype("float64")
        # Converting to double precision destroys the normalization
        T = K / K.sum(axis=1)[:, None]
        return T

    def timescales(self, data_0, data_t, tau):
        T = self.get_transition_matrix(data_0, data_t)
        eigvals = np.linalg.eigvals(T)
        eigvals_sort = np.sort(eigvals)[:-1]  # remove eigenvalue 1
        its = -tau / np.log(np.abs(eigvals_sort[::-1]))

        return its

    def get_transition_matrix_cg(self, data_0, data_t, idx=0):
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            chi_t = net(self.mask(torch.Tensor(data_0).to(self._device)))
            chi_tau = net(self.mask(torch.Tensor(data_t).to(self._device)))
            v, C00, Ctt, C0t, Sigma, u_n = self._ulayer(chi_t, chi_tau, return_u=True)
            _, S_n = self._slayer(v, C00, Ctt, C0t, Sigma, return_S=True)

            for cg_id in range(idx + 1):
                _, chi_t, chi_tau, u_n, S_n, K = self._cg_list[cg_id].get_cg_uS(
                    chi_t, chi_tau, u_n, S_n, return_chi=True, return_K=True
                )

        K = K.to("cpu").numpy().astype("float64")
        # Converting to double precision destroys the normalization
        T = K / K.sum(axis=1)[:, None]
        return T

    def timescales_cg(self, data_0, data_t, tau, idx=0):
        T = self.get_transition_matrix_cg(data_0, data_t, idx=idx)
        eigvals = np.linalg.eigvals(T)
        eigvals_sort = np.sort(eigvals)[:-1]  # remove eigenvalue 1
        its = -tau / np.log(np.abs(eigvals_sort[::-1]))

        return its

    def observables(
        self, data_0, data_t, data_ev=None, data_ac=None, state1=None, state2=None
    ):
        return_mu = False
        return_K = False
        return_S = False
        if data_ev is not None:
            return_mu = True
        if data_ac is not None:
            return_mu = True
            return_K = True
        if state1 is not None:
            return_S = True
        self._lobe.eval()
        net = self._lobe
        with torch.no_grad():
            x_0 = net(self.mask(torch.Tensor(data_0).to(self._device)))
            x_t = net(self.mask(torch.Tensor(data_t).to(self._device)))
            output_u = self._ulayer(x_0, x_t, return_mu=return_mu)
            if return_mu:
                mu = output_u[5]
            Sigma = output_u[4]
            output_S = self._slayer(*output_u[:5], return_K=return_K, return_S=return_S)
            if return_K:
                K = output_S[1]
            if return_S:
                S = output_S[-1]
            ret = []
            if data_ev is not None:
                x_ev = torch.Tensor(data_ev).to(self._device)
                ev_est = obs_ev(x_ev, mu)
                ret.append(ev_est.detach().to("cpu").numpy())
            if data_ac is not None:
                x_ac = torch.Tensor(data_ac).to(self._device)
                ac_est = obs_ac(x_ac, mu, x_t, K, Sigma)
                ret.append(ac_est.detach().to("cpu").numpy())
            if state1 is not None:
                its_est = get_process_eigval(
                    S, Sigma, state1, state2, epsilon=self._epsilon, mode=self._mode
                )
                ret.append(its_est.detach().to("cpu").numpy())
        return ret


class VAMPNet_Multimer_Rev(VAMPNet_Multimer, Transformer, DLEstimatorMixin):
    prefix = "rev_vampnet"

    _MUTABLE_INPUT_DATA = True

    def __init__(
        self,
        multimer: int,
        n_states: int,
        lobe: nn.Module,
        lobe_timelagged: Optional[nn.Module] = None,
        coarse_grain: list = None,
        mask=None,
        device=None,
        optimizer: Union[str, Callable] = "Adam",
        learning_rate: float = 0.0005,
        score_method: str = "VAMP2",
        score_mode: str = "regularize",
        epsilon: float = 0.000001,
        dtype=np.float32,
    ):
        super().__init__(
            multimer,
            n_states,
            lobe,
            lobe_timelagged,
            device,
            optimizer,
            learning_rate,
            score_method,
            score_mode,
            epsilon,
            dtype,
        )
        self.output_dim = n_states * multimer

        self.ulayer = U_layer(
            output_dim=self.output_dim, activation=torch.nn.ReLU()
        ).to(device)
        self.slayer = S_layer(
            output_dim=self.output_dim, activation=torch.nn.ReLU(), renorm=True
        ).to(device)
        self.coarse_grain = coarse_grain
        if self.coarse_grain is not None:
            self.cg_list = []
            self.cg_opt_list = []
            for i, dim_out in enumerate(self.coarse_grain):
                if i == 0:
                    dim_in = self.output_dim
                else:
                    dim_in = self.coarse_grain[i - 1]
                self.cg_list.append(Coarse_grain(dim_in, dim_out).to(device))
                self.cg_opt_list.append(
                    torch.optim.Adam(self.cg_list[-1].parameters(), lr=0.1)
                )
        else:
            self.cg_list = None
        if mask is not None:
            self.mask = mask
            self.optimizer_mask = torch.optim.Adam(
                self.mask.parameters(), lr=self.learning_rate
            )
        else:
            self.mask = torch.nn.Identity()
            self.optimizer_mask = None

        self.setup_optimizer(optimizer, list(self.lobe.parameters()))
        self.optimizer_u = torch.optim.Adam(
            self.ulayer.parameters(), lr=self.learning_rate * 10
        )
        self.optimizer_s = torch.optim.Adam(
            self.slayer.parameters(), lr=self.learning_rate * 100
        )
        self.optimizer_lobe = torch.optim.Adam(
            self.lobe.parameters(), lr=self.learning_rate
        )
        self.optimimzer_all = torch.optim.Adam(
            chain(
                self.ulayer.parameters(),
                self.slayer.parameters(),
                self.lobe.parameters(),
            ),
            lr=self.learning_rate,
        )
        self._train_scores = []
        self._validation_scores = []
        self._train_vampe = []
        self._train_ev = []
        self._train_ac = []
        self._train_its = []
        self._validation_vampe = []
        self._validation_ev = []
        self._validation_ac = []
        self._validation_its = []

    @property
    def train_scores(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores)

    @property
    def train_vampe(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_vampe)

    @property
    def train_ev(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._train_ev).reshape(-1, self._train_ev[0].shape[0])

    @property
    def train_ac(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._train_ac).reshape(-1, self._train_ac[0].shape[0])

    @property
    def train_its(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._train_its).reshape(-1, self._train_its[0].shape[0])

    @property
    def validation_scores(self) -> np.ndarray:
        r"""The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores)

    @property
    def validation_vampe(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_vampe)

    @property
    def validation_ev(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._validation_ev).reshape(
            -1, self._validation_ev[0].shape[0]
        )

    @property
    def validation_ac(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._validation_ac).reshape(
            -1, self._validation_ac[0].shape[0]
        )

    @property
    def validation_its(self) -> np.ndarray:
        r"""The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.concatenate(self._validation_its).reshape(
            -1, self._validation_its[0].shape[0]
        )

    @property
    def epsilon(self) -> float:
        r"""Regularization parameter for matrix inverses.

        :getter: Gets the currently set parameter.
        :setter: Sets a new parameter. Must be non-negative.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        assert value >= 0
        self._epsilon = value

    @property
    def score_method(self) -> str:
        r"""Property which steers the scoring behavior of this estimator.

        :getter: Gets the current score.
        :setter: Sets the score to use.
        :type: str
        """
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        if value not in valid_score_methods:
            raise ValueError(
                f"Tried setting an unsupported scoring method '{value}', "
                f"available are {valid_score_methods}."
            )
        self._score_method = value

    @property
    def lobe(self) -> nn.Module:
        r"""The instantaneous lobe of the VAMPNet.

        :getter: Gets the instantaneous lobe.
        :setter: Sets a new lobe.
        :type: torch.nn.Module
        """
        return self._lobe

    @lobe.setter
    def lobe(self, value: nn.Module):
        self._lobe = value
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
        else:
            self._lobe = self._lobe.double()
        self._lobe = self._lobe.to(device=self.device)

    def forward(self, data):
        if data.get_device():
            data = data.to(device=self.device)

        return self.lobe(self.mask(data))

    def reset_scores(self):
        self._train_scores = []
        self._validation_scores = []

    def partial_fit(
        self,
        data,
        mask: bool = False,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
    ):
        r"""Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        with disable_TF32():
            if self.dtype == np.float32:
                self._lobe = self._lobe.float()
            elif self.dtype == np.float64:
                self._lobe = self._lobe.double()

            self.lobe.train()
            assert isinstance(data, (list, tuple)) and len(data) == 2, (
                "Data must be a list or tuple of batches belonging to instantaneous "
                "and respective time-lagged data."
            )

            batch_0, batch_t = data[0], data[1]

            if isinstance(data[0], np.ndarray):
                batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(
                    device=self.device
                )
            if isinstance(data[1], np.ndarray):
                batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(
                    device=self.device
                )

            self.optimizer_lobe.zero_grad()
            self.optimizer_u.zero_grad()
            self.optimizer_s.zero_grad()
            if self.optimizer_mask is not None and mask:
                self.optimizer_mask.zero_grad()
            x_0 = self.forward(batch_0)
            x_t = self.forward(batch_t)

            loss_value = vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[0]

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                chain(
                    self.lobe.parameters(),
                    self.mask.parameters(),
                    self.ulayer.parameters(),
                    self.slayer.parameters(),
                ),
                CLIP_VALUE,
            )
            if self.mask is not None and mask:
                self.optimizer_mask.step()
            self.optimizer_lobe.step()
            self.optimizer_u.step()
            self.optimizer_s.step()

            if train_score_callback is not None:
                lval_detached = loss_value.detach()
                train_score_callback(self._step, -lval_detached)
            if tb_writer is not None:
                tb_writer.add_scalars("Loss", {"train": loss_value.item()}, self._step)
                tb_writer.add_scalars(
                    "VAMPE", {"train": -loss_value.item()}, self._step
                )
            self._train_scores.append((self._step, (-loss_value).item()))
            self._step += 1

            return self

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
        r"""Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        with disable_TF32():
            self.lobe.eval()

            with torch.no_grad():
                val = self.forward(validation_data[0])
                val_t = self.forward(validation_data[1])
                score_value = -vampe_loss_rev(val, val_t, self.ulayer, self.slayer)[0]
                return score_value

    def fit(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs=1,
        validation_loader=None,
        train_mode="all",
        mask=False,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        validation_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
        progress=None,
        **kwargs,
    ):
        r"""Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training for lobe, u, and s
            'us' : training for u and s
            's'  : training for s
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        from deeptime.util.platform import handle_progress_bar

        progress = handle_progress_bar(progress)

        with disable_TF32():
            self._step = 0

            # and train
            if train_mode == "all":
                print("Train all")
                for epoch in progress(
                    range(n_epochs),
                    desc="Train all VAMPNet epoch",
                    total=n_epochs,
                    leave=False,
                ):
                    for batch_0, batch_t in data_loader:
                        self.partial_fit(
                            (batch_0, batch_t),
                            mask=mask,
                            train_score_callback=train_score_callback,
                            tb_writer=tb_writer,
                        )
                    if validation_loader is not None:
                        with torch.no_grad():
                            scores = []
                            for val_batch in validation_loader:
                                scores.append(
                                    self.validate((val_batch[0], val_batch[1]))
                                )
                            mean_score = torch.mean(torch.stack(scores))
                            print(mean_score)
                            self._validation_scores.append(
                                (self._step, mean_score.item())
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalars(
                                    "Loss", {"valid": -mean_score.item()}, self._step
                                )
                                tb_writer.add_scalars(
                                    "VAMPE", {"valid": mean_score.item()}, self._step
                                )
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, mean_score)
            else:
                chi_t, chi_tau = [], []
                with torch.no_grad():
                    for batch_0, batch_t in data_loader:
                        chi_t.append(self.forward(batch_0).detach())
                        chi_tau.append(self.forward(batch_t).detach())
                    x_0 = torch.cat(chi_t, dim=0)
                    x_t = torch.cat(chi_tau, dim=0)
                    if validation_loader is not None:
                        chi_val_t, chi_val_tau = [], []
                        for batch_0, batch_t in validation_loader:
                            chi_val_t.append(self.forward(batch_0).detach())
                            chi_val_tau.append(self.forward(batch_t).detach())
                        x_val_0 = torch.cat(chi_val_t, dim=0)
                        x_val_t = torch.cat(chi_val_tau, dim=0)
                if train_mode == "us" or train_mode == "u":
                    print("Train u/s")

                    for epoch in progress(
                        range(n_epochs),
                        desc="Train u/s VAMPNet epoch",
                        total=n_epochs,
                        leave=False,
                    ):
                        self.optimizer_u.zero_grad()
                        if train_mode == "us":
                            self.optimizer_s.zero_grad()

                        loss_value = vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[
                            0
                        ]
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            chain(self.ulayer.parameters(), self.slayer.parameters()),
                            CLIP_VALUE,
                        )
                        self.optimizer_u.step()
                        if train_mode == "us":
                            self.optimizer_s.step()

                        if train_score_callback is not None:
                            lval_detached = loss_value.detach()
                            train_score_callback(self._step, -lval_detached)
                        self._train_scores.append((self._step, (-loss_value).item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars(
                                "Loss", {"train": loss_value.item()}, self._step
                            )
                            tb_writer.add_scalars(
                                "VAMPE", {"train": -loss_value.item()}, self._step
                            )
                        if validation_loader is not None:
                            with torch.no_grad():
                                score_val = -vampe_loss_rev(
                                    x_val_0, x_val_t, self.ulayer, self.slayer
                                )[0]
                                self._validation_scores.append(
                                    (self._step, score_val.item())
                                )
                                if tb_writer is not None:
                                    tb_writer.add_scalars(
                                        "Loss", {"valid": -score_val.item()}, self._step
                                    )
                                    tb_writer.add_scalars(
                                        "VAMPE", {"valid": score_val.item()}, self._step
                                    )
                                if validation_score_callback is not None:
                                    validation_score_callback(self._step, score_val)
                        self._step += 1
                if train_mode == "s":
                    print("Train s")

                    with torch.no_grad():
                        v, C_00, C_11, C_01, Sigma = self.ulayer(x_0, x_t)
                        v_val, C_00_val, C_11_val, C_01_val, Sigma_val = self.ulayer(
                            x_val_0, x_val_t
                        )
                    for epoch in progress(
                        range(n_epochs),
                        desc="Train s VAMPNet epoch",
                        total=n_epochs,
                        leave=False,
                    ):
                        self.optimizer_s.zero_grad()

                        loss_value = vampe_loss_rev_only_S(
                            v, C_00, C_11, C_01, Sigma, self.slayer
                        )[0]
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.slayer.parameters(), CLIP_VALUE
                        )
                        self.optimizer_s.step()

                        if train_score_callback is not None:
                            lval_detached = loss_value.detach()
                            train_score_callback(self._step, -lval_detached)
                        self._train_scores.append((self._step, (-loss_value).item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars(
                                "Loss", {"train": -loss_value.item()}, self._step
                            )
                            tb_writer.add_scalars(
                                "VAMPE", {"train": loss_value.item()}, self._step
                            )
                        if validation_loader is not None:
                            with torch.no_grad():
                                score_val = -vampe_loss_rev_only_S(
                                    v_val,
                                    C_00_val,
                                    C_11_val,
                                    C_01_val,
                                    Sigma_val,
                                    self.slayer,
                                )[0]
                                self._validation_scores.append(
                                    (self._step, score_val.item())
                                )
                                if tb_writer is not None:
                                    tb_writer.add_scalars(
                                        "Loss", {"valid": -score_val.item()}, self._step
                                    )
                                    tb_writer.add_scalars(
                                        "VAMPE", {"valid": score_val.item()}, self._step
                                    )
                                if validation_score_callback is not None:
                                    validation_score_callback(self._step, score_val)
                        self._step += 1

            return self

    def fit_routine(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs=1,
        validation_loader=None,
        rel=1e-4,
        reset_u=False,
        max_iter=100,
        mask=False,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        validation_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
        progress=None,
        **kwargs,
    ):
        r"""Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training for lobe, u, and s
            'us' : training for u and s
            's'  : training for s
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        from deeptime.util.platform import handle_progress_bar

        progress = handle_progress_bar(progress)

        with disable_TF32():
            self._step = 0

            # and train
            for g in self.optimizer_lobe.param_groups:
                lr_chi = g["lr"]
                g["lr"] = lr_chi / 10
            if self.optimizer_mask is not None:
                for g in self.optimizer_mask.param_groups:
                    lr_mask = g["lr"]
                    g["lr"] = lr_mask / 10
            for g in self.optimizer_u.param_groups:
                lr_u = g["lr"]
            for g in self.optimizer_s.param_groups:
                lr_s = g["lr"]
            for epoch in progress(
                range(n_epochs),
                desc="Train only VAMPNet epoch",
                total=n_epochs,
                leave=False,
            ):
                for g in self.optimizer_u.param_groups:
                    g["lr"] = lr_u / 10
                for g in self.optimizer_s.param_groups:
                    g["lr"] = lr_s / 10
                for batch_0, batch_t in data_loader:
                    self.partial_fit(
                        (batch_0, batch_t),
                        mask=mask,
                        train_score_callback=train_score_callback,
                        tb_writer=tb_writer,
                    )

                chi_t, chi_tau = [], []
                with torch.no_grad():
                    for batch_0, batch_t in data_loader:
                        chi_t.append(self.forward(batch_0).detach())
                        chi_tau.append(self.forward(batch_t).detach())
                    x_0 = torch.cat(chi_t, dim=0)
                    x_t = torch.cat(chi_tau, dim=0)
                    score_value_before = -vampe_loss_rev(
                        x_0, x_t, self.ulayer, self.slayer
                    )[0].detach()
                flag = True
                # reduce the learning rate of u and S
                for g in self.optimizer_u.param_groups:
                    g["lr"] = lr_u / 2
                for g in self.optimizer_s.param_groups:
                    g["lr"] = lr_s / 2
                counter = 0
                #             print('Score before loop', score_value_before.item())
                if reset_u:
                    cov_00, cov_0t, cov_tt = covariances(x_0, x_t, remove_mean=False)
                    cov_00_inv = (
                        sym_inverse(cov_00, epsilon=self.epsilon, mode=self.score_mode)
                        .to("cpu")
                        .numpy()
                    )

                    K_vamp = cov_00_inv @ cov_0t.to("cpu").numpy()

                    # estimate pi, the stationary distribution vector
                    eigv, eigvec = np.linalg.eig(K_vamp.T)
                    ind_pi = np.argmin((eigv - 1) ** 2)

                    pi_vec = np.real(eigvec[:, ind_pi])
                    pi = pi_vec / np.sum(pi_vec, keepdims=True)
                    #                 print('pi', pi)
                    # reverse the consruction of u
                    u_optimal = cov_00_inv @ pi
                    #                 print('u optimal', u_optimal)

                    # u_kernel = np.log(np.exp(np.abs(u_optimal))-1) # if softplus
                    # for relu
                    u_kernel = np.abs(u_optimal)

                    with torch.no_grad():
                        for param in self.ulayer.parameters():
                            param.copy_(torch.Tensor(u_kernel[None, :]))
                while flag:
                    self.optimizer_u.zero_grad()
                    self.optimizer_s.zero_grad()

                    score = -vampe_loss_rev(x_0, x_t, self.ulayer, self.slayer)[0]
                    print(score)
                    loss_value = -score
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        chain(self.ulayer.parameters(), self.slayer.parameters()),
                        CLIP_VALUE,
                    )
                    self.optimizer_u.step()
                    self.optimizer_s.step()
                    if (score - score_value_before) < rel and counter > 0:
                        flag = False
                    counter += 1
                    if counter > max_iter:
                        #                     print('Reached max number of iterations')
                        flag = False
                    score_value_before = score
                #             print('and after: ', score.item())
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        for val_batch in validation_loader:
                            scores.append(self.validate((val_batch[0], val_batch[1])))
                        mean_score = torch.mean(torch.stack(scores))
                        self._validation_scores.append((self._step, mean_score.item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars(
                                "Loss", {"valid": -mean_score.item()}, self._step
                            )
                            tb_writer.add_scalars(
                                "VAMPE", {"valid": mean_score.item()}, self._step
                            )
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, mean_score)
            for g in self.optimizer_lobe.param_groups:
                g["lr"] = lr_chi
            if self.optimizer_mask is not None:
                for g in self.optimizer_mask.param_groups:
                    g["lr"] = lr_mask
            return self

    def fit_cg(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs=1,
        validation_loader=None,
        train_mode="single",
        idx=0,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        validation_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
        progress=None,
        **kwargs,
    ):
        r"""Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training u, and s and all coarse graining matrices
            'single' : training for coarse graining matrix of layer idx
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        from deeptime.util.platform import handle_progress_bar

        progress = handle_progress_bar(progress)
        with disable_TF32():
            self._step = 0

            # and train
            chi_t, chi_tau = [], []
            with torch.no_grad():
                for batch_0, batch_t in data_loader:
                    chi_t.append(self.forward(batch_0).detach())
                    chi_tau.append(self.forward(batch_t).detach())
                chi_t = torch.cat(chi_t, dim=0)
                chi_tau = torch.cat(chi_tau, dim=0)
                if validation_loader is not None:
                    chi_val_t, chi_val_tau = [], []
                    for batch_0, batch_t in validation_loader:
                        chi_val_t.append(self.forward(batch_0).detach())
                        chi_val_tau.append(self.forward(batch_t).detach())
                    chi_val_t = torch.cat(chi_val_t, dim=0)
                    chi_val_tau = torch.cat(chi_val_tau, dim=0)
            if train_mode == "all":
                for epoch in progress(
                    range(n_epochs),
                    desc="Train all VAMPNet epoch",
                    total=n_epochs,
                    leave=False,
                ):
                    self.optimizer_u.zero_grad()
                    self.optimizer_s.zero_grad()
                    for opt in self.cg_opt_list:
                        opt.zero_grad()

                    v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(
                        chi_t, chi_tau, return_u=True
                    )
                    matrix, S_n = self.slayer(v, C00, Ctt, C0t, Sigma, return_S=True)

                    chi_t_n, chi_tau_n = chi_t, chi_tau
                    loss_value = torch.trace(matrix)
                    for cg_id in range(len(self.coarse_grain)):
                        matrix_cg, chi_t_n, chi_tau_n, u_n, S_n = self.cg_list[
                            cg_id
                        ].get_cg_uS(chi_t_n, chi_tau_n, u_n, S_n, return_chi=True)
                        loss_value += torch.trace(matrix_cg)

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        chain(self.ulayer.parameters(), self.slayer.parameters()),
                        CLIP_VALUE,
                    )
                    for lay_cg in self.cg_list:
                        torch.nn.utils.clip_grad_norm_(lay_cg.parameters(), CLIP_VALUE)
                    self.optimizer_u.step()
                    self.optimizer_s.step()
                    for opt in self.cg_opt_list:
                        opt.step()

                    if train_score_callback is not None:
                        lval_detached = loss_value.detach()
                        train_score_callback(self._step, -lval_detached)
                    self._train_scores.append((self._step, (-loss_value).item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars(
                            "Loss", {"cg_train": loss_value.item()}, self._step
                        )
                        tb_writer.add_scalars(
                            "VAMPE", {"cg_train": -loss_value.item()}, self._step
                        )

                    if validation_loader is not None:
                        with torch.no_grad():
                            v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(
                                chi_val_t, chi_val_tau, return_u=True
                            )
                            matrix, S_n = self.slayer(
                                v, C00, Ctt, C0t, Sigma, return_S=True
                            )
                            chi_val_t_n, chi_val_tau_n = chi_val_t, chi_val_tau
                            loss_value = torch.trace(matrix)
                            for cg_id in range(len(self.coarse_grain)):
                                (
                                    matrix_cg,
                                    chi_val_t_n,
                                    chi_val_tau_n,
                                    u_n,
                                    S_n,
                                ) = self.cg_list[cg_id].get_cg_uS(
                                    chi_val_t_n,
                                    chi_val_tau_n,
                                    u_n,
                                    S_n,
                                    return_chi=True,
                                )
                                loss_value += torch.trace(matrix_cg)
                            score_val = -loss_value
                            self._validation_scores.append(
                                (self._step, score_val.item())
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalars(
                                    "Loss", {"cg_valid": -score_val.item()}, self._step
                                )
                                tb_writer.add_scalars(
                                    "VAMPE", {"cg_valid": score_val.item()}, self._step
                                )
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, score_val)
                    self._step += 1
            elif train_mode == "single":
                with torch.no_grad():
                    v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(
                        chi_t, chi_tau, return_u=True
                    )
                    _, S_n = self.slayer(v, C00, Ctt, C0t, Sigma, return_S=True)

                    if idx > 0:
                        for cg_id in range(idx):
                            _, chi_t, chi_tau, u_n, S_n = self.cg_list[cg_id].get_cg_uS(
                                chi_t, chi_tau, u_n, S_n, return_chi=True
                            )
                    if validation_loader is not None:
                        (
                            v_val,
                            C00_val,
                            Ctt_val,
                            C0t_val,
                            Sigma_val,
                            u_n_val,
                        ) = self.ulayer(chi_val_t, chi_val_tau, return_u=True)
                        _, S_n_val = self.slayer(
                            v_val, C00_val, Ctt_val, C0t_val, Sigma_val, return_S=True
                        )
                        for cg_id in range(idx):
                            _, chi_val_t, chi_val_tau, u_n_val, S_n_val = self.cg_list[
                                cg_id
                            ].get_cg_uS(
                                chi_val_t,
                                chi_val_tau,
                                u_n_val,
                                S_n_val,
                                return_chi=True,
                            )
                for epoch in progress(
                    range(n_epochs),
                    desc="Train single VAMPNet epoch",
                    total=n_epochs,
                    leave=False,
                ):
                    self.cg_opt_list[idx].zero_grad()
                    matrix_cg = self.cg_list[idx].get_cg_uS(
                        chi_t, chi_tau, u_n, S_n, return_chi=False
                    )[0]
                    loss_value = torch.trace(matrix_cg)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.cg_list[idx].parameters(), CLIP_VALUE
                    )
                    self.cg_opt_list[idx].step()

                    if train_score_callback is not None:
                        lval_detached = loss_value.detach()
                        train_score_callback(self._step, -lval_detached)
                    self._train_scores.append((self._step, (-loss_value).item()))
                    if tb_writer is not None:
                        tb_writer.add_scalars(
                            "Loss", {"cg_train": loss_value.item()}, self._step
                        )
                        tb_writer.add_scalars(
                            "VAMPE", {"cg_train": -loss_value.item()}, self._step
                        )
                    if validation_loader is not None:
                        with torch.no_grad():
                            matrix_cg = self.cg_list[idx].get_cg_uS(
                                chi_val_t,
                                chi_val_tau,
                                u_n_val,
                                S_n_val,
                                return_chi=False,
                            )[0]

                            score_val = -torch.trace(matrix_cg)

                            self._validation_scores.append(
                                (self._step, score_val.item())
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalars(
                                    "Loss", {"cg_valid": score_val.item()}, self._step
                                )
                                tb_writer.add_scalars(
                                    "VAMPE", {"cg_valid": -score_val.item()}, self._step
                                )
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, score_val)

                    self._step += 1

            return self

    def partial_fit_obs(
        self,
        data,
        data_ev,
        data_ac,
        exp_ev=None,
        exp_ac=None,
        exp_its=None,
        xi_ev=None,
        xi_ac=None,
        xi_its=None,
        its_state1=None,
        its_state2=None,
        mask=False,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
    ):
        r"""Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        with disable_TF32():
            if self.dtype == np.float32:
                self._lobe = self._lobe.float()
            elif self.dtype == np.float64:
                self._lobe = self._lobe.double()

            self.lobe.train()
            assert isinstance(data, (list, tuple)) and len(data) == 2, (
                "Data must be a list or tuple of batches belonging to instantaneous "
                "and respective time-lagged data."
            )

            batch_0, batch_t = data[0], data[1]

            if isinstance(data[0], np.ndarray):
                batch_0 = torch.from_numpy(data[0].astype(self.dtype))
            if isinstance(data[1], np.ndarray):
                batch_t = torch.from_numpy(data[1].astype(self.dtype))

            return_mu = False
            return_K = False
            return_Sigma = False
            return_S = False

            if exp_ev is not None:
                return_mu = True
                batch_ev = data_ev
                if isinstance(data_ev, np.ndarray):
                    batch_ev = torch.from_numpy(data_ev.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(exp_ev, np.ndarray):
                    exp_ev = torch.from_numpy(exp_ev.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(xi_ev, np.ndarray):
                    xi_ev = torch.from_numpy(xi_ev.astype(self.dtype)).to(
                        device=self.device
                    )
            if exp_ac is not None:
                return_mu = True
                return_K = True
                return_Sigma = True
                batch_ac = data_ac
                if isinstance(data_ac, np.ndarray):
                    batch_ac = torch.from_numpy(data_ac.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(exp_ac, np.ndarray):
                    exp_ac = torch.from_numpy(exp_ac.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(xi_ac, np.ndarray):
                    xi_ac = torch.from_numpy(xi_ac.astype(self.dtype)).to(
                        device=self.device
                    )
            if exp_its is not None:
                return_S = True
                return_Sigma = True
                if isinstance(exp_its, np.ndarray):
                    exp_its = torch.from_numpy(exp_its.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(xi_its, np.ndarray):
                    xi_its = torch.from_numpy(xi_its.astype(self.dtype)).to(
                        device=self.device
                    )

            self.optimizer_lobe.zero_grad()
            self.optimizer_u.zero_grad()
            self.optimizer_s.zero_grad()
            if mask and self.optimizer_mask is not None:
                self.optimizer_mask.zero_grad()
            x_0 = self.forward(batch_0)
            x_t = self.forward(batch_t)

            output_loss = vampe_loss_rev(
                x_0,
                x_t,
                self.ulayer,
                self.slayer,
                return_mu=return_mu,
                return_Sigma=return_Sigma,
                return_K=return_K,
                return_S=return_S,
            )
            vampe_loss = output_loss[0]
            loss_value = -vampe_loss  # vampe loss
            counter = 1
            if return_K:
                K = output_loss[counter]
                counter += 1
            if return_S:
                S = output_loss[counter]
                counter += 1
            if return_mu:
                mu = output_loss[counter]
            if return_Sigma:
                Sigma = output_loss[-1]
            if exp_ev is not None:
                loss_ev, est_ev = obs_ev_loss(batch_ev, mu, exp_ev, xi_ev)
                loss_value += loss_ev
            if exp_ac is not None:
                loss_ac, est_ac = obs_ac_loss(
                    batch_ac, mu, x_t, K, Sigma, exp_ac, xi_ac
                )
                loss_value += loss_ac
            if exp_its is not None:
                loss_its, est_its = obs_its_loss(
                    S,
                    Sigma,
                    its_state1,
                    its_state2,
                    exp_its,
                    xi_its,
                    epsilon=self.epsilon,
                    mode=self.score_mode,
                )
                loss_value += loss_its
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                chain(
                    self.lobe.parameters(),
                    self.mask.parameters(),
                    self.ulayer.parameters(),
                    self.slayer.parameters(),
                ),
                CLIP_VALUE,
            )
            self.optimizer_lobe.step()
            self.optimizer_u.step()
            self.optimizer_s.step()
            if mask and self.optimizer_mask is not None:
                self.optimizer_mask.step()

            if train_score_callback is not None:
                lval_detached = loss_value.detach()
                train_score_callback(self._step, lval_detached)
            self._train_scores.append((self._step, (loss_value).item()))
            self._train_vampe.append((self._step, (vampe_loss).item()))
            if tb_writer is not None:
                tb_writer.add_scalars("Loss", {"train": loss_value.item()}, self._step)
                tb_writer.add_scalars("VAMPE", {"train": vampe_loss.item()}, self._step)
            if exp_ev is not None:
                self._train_ev.append(
                    np.concatenate(([self._step], (est_ev).detach().to("cpu").numpy()))
                )
                if tb_writer is not None:
                    for i in range(est_ev.shape[0]):
                        tb_writer.add_scalars(
                            "EV", {"train_" + str(i + 1): est_ev[i].item()}, self._step
                        )
            if exp_ac is not None:
                self._train_ac.append(
                    np.concatenate(([self._step], (est_ac).detach().to("cpu").numpy()))
                )
                if tb_writer is not None:
                    for i in range(est_ac.shape[0]):
                        tb_writer.add_scalars(
                            "AC", {"train_" + str(i + 1): est_ac[i].item()}, self._step
                        )
            if exp_its is not None:
                self._train_its.append(
                    np.concatenate(([self._step], (est_its).detach().to("cpu").numpy()))
                )
                if tb_writer is not None:
                    for i in range(est_its.shape[0]):
                        tb_writer.add_scalars(
                            "ITS",
                            {"train_" + str(i + 1): est_its[i].item()},
                            self._step,
                        )
            self._step += 1

            return self

    def validate_obs(
        self,
        validation_data: Tuple[torch.Tensor],
        val_data_ev=None,
        val_data_ac=None,
        exp_ev=None,
        exp_ac=None,
        exp_its=None,
        xi_ev=None,
        xi_ac=None,
        xi_its=None,
        its_state1=None,
        its_state2=None,
    ) -> torch.Tensor:
        r"""Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        with disable_TF32():
            self.lobe.eval()
        return_mu = False
        return_K = False
        return_Sigma = False
        return_S = False

        if exp_ev is not None:
            return_mu = True
            batch_ev = val_data_ev
            if isinstance(val_data_ev, np.ndarray):
                batch_ev = torch.from_numpy(val_data_ev.astype(self.dtype)).to(
                    device=self.device
                )
            if isinstance(exp_ev, np.ndarray):
                exp_ev = torch.from_numpy(exp_ev.astype(self.dtype)).to(
                    device=self.device
                )
            if isinstance(xi_ev, np.ndarray):
                xi_ev = torch.from_numpy(xi_ev.astype(self.dtype)).to(
                    device=self.device
                )
        if exp_ac is not None:
            return_mu = True
            return_K = True
            return_Sigma = True
            batch_ac = val_data_ac
            if isinstance(val_data_ac, np.ndarray):
                batch_ac = torch.from_numpy(val_data_ac.astype(self.dtype)).to(
                    device=self.device
                )
            if isinstance(exp_ac, np.ndarray):
                exp_ac = torch.from_numpy(exp_ac.astype(self.dtype)).to(
                    device=self.device
                )
            if isinstance(xi_ac, np.ndarray):
                xi_ac = torch.from_numpy(xi_ac.astype(self.dtype)).to(
                    device=self.device
                )
        if exp_its is not None:
            return_S = True
            return_Sigma = True
            if isinstance(exp_its, np.ndarray):
                exp_its = torch.from_numpy(exp_its.astype(self.dtype)).to(
                    device=self.device
                )
            if isinstance(xi_its, np.ndarray):
                xi_its = torch.from_numpy(xi_its.astype(self.dtype)).to(
                    device=self.device
                )
        with torch.no_grad():
            val = self.forward(validation_data[0])
            val_t = self.forward(validation_data[1])
            output_loss = vampe_loss_rev(
                val,
                val_t,
                self.ulayer,
                self.slayer,
                return_mu=return_mu,
                return_Sigma=return_Sigma,
                return_K=return_K,
                return_S=return_S,
            )
            vampe_loss = output_loss[0]
            score_value = -vampe_loss  # vampe loss
            ret = [score_value, vampe_loss]
            counter = 1
            if return_K:
                K = output_loss[counter]
                counter += 1
            if return_S:
                S = output_loss[counter]
                counter += 1
            if return_mu:
                mu = output_loss[counter]
            if return_Sigma:
                Sigma = output_loss[-1]
            if exp_ev is not None:
                loss_ev, est_ev = obs_ev_loss(batch_ev, mu, exp_ev, xi_ev)
                score_value += loss_ev
                ret.append(est_ev)
            if exp_ac is not None:
                loss_ac, est_ac = obs_ac_loss(
                    batch_ac, mu, val_t, K, Sigma, exp_ac, xi_ac
                )
                score_value += loss_ac
                ret.append(est_ac)
            if exp_its is not None:
                loss_its, est_its = obs_its_loss(
                    S,
                    Sigma,
                    its_state1,
                    its_state2,
                    exp_its,
                    xi_its,
                    epsilon=self.epsilon,
                    mode=self.score_mode,
                )
                score_value += loss_its
                ret.append(est_its)
            return ret

    def fit_obs(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs=1,
        validation_loader=None,
        train_mode="all",
        exp_ev=None,
        exp_ac=None,
        exp_its=None,
        xi_ev=None,
        xi_ac=None,
        xi_its=None,
        its_state1=None,
        its_state2=None,
        mask=False,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        validation_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
        progress=None,
        **kwargs,
    ):
        r"""Fits a VampNet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_mode : str, default='all'
            'all': training for lobe, u, and s
            'us' : training for u and s
            's'  : training for s
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        from deeptime.util.platform import handle_progress_bar

        progress = handle_progress_bar(progress)
        with disable_TF32():
            self._step = 0
            if exp_ev is not None:
                if isinstance(exp_ev, list):
                    exp_ev = np.array(exp_ev)
                if isinstance(xi_ev, list):
                    xi_ev = np.array(xi_ev)
                if isinstance(exp_ev, np.ndarray):
                    exp_ev = torch.from_numpy(exp_ev.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(xi_ev, np.ndarray):
                    xi_ev = torch.from_numpy(xi_ev.astype(self.dtype)).to(
                        device=self.device
                    )
            if exp_ac is not None:
                if isinstance(exp_ac, list):
                    exp_ac = np.array(exp_ac)
                if isinstance(xi_ac, list):
                    xi_ac = np.array(xi_ac)
                if isinstance(exp_ac, np.ndarray):
                    exp_ac = torch.from_numpy(exp_ac.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(xi_ac, np.ndarray):
                    xi_ac = torch.from_numpy(xi_ac.astype(self.dtype)).to(
                        device=self.device
                    )
            if exp_its is not None:
                if isinstance(exp_its, list):
                    exp_its = np.array(exp_its)
                if isinstance(xi_its, list):
                    xi_its = np.array(xi_its)
                if isinstance(exp_its, np.ndarray):
                    exp_its = torch.from_numpy(exp_its.astype(self.dtype)).to(
                        device=self.device
                    )
                if isinstance(xi_its, np.ndarray):
                    xi_its = torch.from_numpy(xi_its.astype(self.dtype)).to(
                        device=self.device
                    )
            # and train
            if train_mode == "all":
                for epoch in progress(
                    range(n_epochs),
                    desc="Train all VAMPNet epoch",
                    total=n_epochs,
                    leave=False,
                ):
                    for batch in data_loader:
                        batch_0, batch_t = batch[0], batch[1]
                        if exp_ev is not None:
                            batch_ev = batch[2].to(device=self.device)
                        else:
                            batch_ev = None
                        if exp_ac is not None:
                            batch_ac = batch[-1].to(device=self.device)
                        else:
                            batch_ac = None
                        self.partial_fit_obs(
                            (batch_0, batch_t),
                            data_ev=batch_ev,
                            data_ac=batch_ac,
                            exp_ev=exp_ev,
                            exp_ac=exp_ac,
                            exp_its=exp_its,
                            xi_ev=xi_ev,
                            xi_ac=xi_ac,
                            xi_its=xi_its,
                            its_state1=its_state1,
                            its_state2=its_state2,
                            mask=mask,
                            train_score_callback=train_score_callback,
                            tb_writer=tb_writer,
                        )
                    if validation_loader is not None:
                        with torch.no_grad():
                            scores = []
                            scores_vampe = []
                            idx_ac = 2
                            if exp_ev is not None:
                                scores_ev = []
                                idx_ac = 3
                            if exp_ac is not None:
                                scores_ac = []
                            if exp_its is not None:
                                scores_its = []
                            for val_batch in validation_loader:
                                if exp_ev is not None:
                                    data_val_ev = val_batch[2].to(device=self.device)
                                else:
                                    data_val_ev = None
                                if exp_ac is not None:
                                    data_val_ac = val_batch[-1].to(device=self.device)
                                else:
                                    data_val_ac = None
                                all_scores = self.validate_obs(
                                    (
                                        val_batch[0].to(device=self.device),
                                        val_batch[1].to(device=self.device),
                                    ),
                                    val_data_ev=data_val_ev,
                                    val_data_ac=data_val_ac,
                                    exp_ev=exp_ev,
                                    exp_ac=exp_ac,
                                    exp_its=exp_its,
                                    xi_ev=xi_ev,
                                    xi_ac=xi_ac,
                                    xi_its=xi_its,
                                    its_state1=its_state1,
                                    its_state2=its_state2,
                                )
                                scores.append(all_scores[0])
                                scores_vampe.append(all_scores[1])
                                if exp_ev is not None:
                                    scores_ev.append(all_scores[2])
                                if exp_ac is not None:
                                    scores_ac.append(all_scores[idx_ac])
                                if exp_its is not None:
                                    scores_its.append(all_scores[-1])

                            mean_score = torch.mean(torch.stack(scores))
                            self._validation_scores.append(
                                (self._step, mean_score.item())
                            )
                            mean_vampe = torch.mean(torch.stack(scores_vampe))
                            self._validation_vampe.append(
                                (self._step, mean_vampe.item())
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalars(
                                    "Loss", {"valid": mean_score.item()}, self._step
                                )
                                tb_writer.add_scalars(
                                    "VAMPE", {"valid": mean_vampe.item()}, self._step
                                )
                            if exp_ev is not None:
                                mean_ev = torch.mean(torch.stack(scores_ev), dim=0)
                                self._validation_ev.append(
                                    np.concatenate(
                                        (
                                            [self._step],
                                            (mean_ev).detach().to("cpu").numpy(),
                                        )
                                    )
                                )
                                if tb_writer is not None:
                                    for i in range(mean_ev.shape[0]):
                                        tb_writer.add_scalars(
                                            "EV",
                                            {"valid_" + str(i + 1): mean_ev[i].item()},
                                            self._step,
                                        )
                            if exp_ac is not None:
                                mean_ac = torch.mean(torch.stack(scores_ac), dim=0)
                                self._validation_ac.append(
                                    np.concatenate(
                                        (
                                            [self._step],
                                            (mean_ac).detach().to("cpu").numpy(),
                                        )
                                    )
                                )
                                if tb_writer is not None:
                                    for i in range(mean_ac.shape[0]):
                                        tb_writer.add_scalars(
                                            "AC",
                                            {"valid_" + str(i + 1): mean_ac[i].item()},
                                            self._step,
                                        )
                            if exp_its is not None:
                                mean_its = torch.mean(torch.stack(scores_its), dim=0)
                                self._validation_its.append(
                                    np.concatenate(
                                        (
                                            [self._step],
                                            (mean_its).detach().to("cpu").numpy(),
                                        )
                                    )
                                )
                                if tb_writer is not None:
                                    for i in range(mean_its.shape[0]):
                                        tb_writer.add_scalars(
                                            "ITS",
                                            {"valid_" + str(i + 1): mean_its[i].item()},
                                            self._step,
                                        )
                            if validation_score_callback is not None:
                                validation_score_callback(self._step, mean_score)
            else:
                return_mu = False
                return_K = False
                return_Sigma = False
                return_S = False
                chi_t, chi_tau = [], []
                if exp_ev is not None:
                    data_ev = []
                    return_mu = True
                if exp_ac is not None:
                    data_ac = []
                    return_mu = True
                    return_K = True
                    return_Sigma = True
                if exp_its is not None:
                    return_S = True
                    return_Sigma = True
                with torch.no_grad():
                    for batch in data_loader:
                        batch_0, batch_t = batch[0], batch[1]
                        chi_t.append(self.forward(batch_0).detach())
                        chi_tau.append(self.forward(batch_t).detach())
                        if exp_ev is not None:
                            data_ev.append(batch[2].to(device=self.device))
                        if exp_ac is not None:
                            data_ac.append(batch[-1].to(device=self.device))
                    x_0 = torch.cat(chi_t, dim=0)
                    x_t = torch.cat(chi_tau, dim=0)
                    if exp_ev is not None:
                        x_ev = torch.cat(data_ev, dim=0)
                    if exp_ac is not None:
                        x_ac = torch.cat(data_ac, dim=0)
                    if validation_loader is not None:
                        chi_val_t, chi_val_tau = [], []
                        if exp_ev is not None:
                            data_val_ev = []
                        if exp_ac is not None:
                            data_val_ac = []
                        for batch in validation_loader:
                            batch_0, batch_t = batch[0], batch[1]
                            chi_val_t.append(self.forward(batch_0).detach())
                            chi_val_tau.append(self.forward(batch_t).detach())
                            if exp_ev is not None:
                                data_val_ev.append(batch[2].to(device=self.device))
                            if exp_ac is not None:
                                data_val_ac.append(batch[-1].to(device=self.device))
                        x_val_0 = torch.cat(chi_val_t, dim=0)
                        x_val_t = torch.cat(chi_val_tau, dim=0)
                        if exp_ev is not None:
                            x_val_ev = torch.cat(data_val_ev, dim=0)
                        if exp_ac is not None:
                            x_val_ac = torch.cat(data_val_ac, dim=0)
                if train_mode == "us" or train_mode == "u":
                    for epoch in progress(
                        range(n_epochs),
                        desc="Train u/s VAMPNet epoch",
                        total=n_epochs,
                        leave=False,
                    ):
                        self.optimizer_u.zero_grad()
                        if train_mode == "us":
                            self.optimizer_s.zero_grad()

                        output_loss = vampe_loss_rev(
                            x_0,
                            x_t,
                            self.ulayer,
                            self.slayer,
                            return_mu=return_mu,
                            return_Sigma=return_Sigma,
                            return_K=return_K,
                            return_S=return_S,
                        )
                        vampe_loss = output_loss[0]
                        loss_value = -vampe_loss  # vampe loss
                        counter = 1
                        if return_K:
                            K = output_loss[counter]
                            counter += 1
                        if return_S:
                            S = output_loss[counter]
                            counter += 1
                        if return_mu:
                            mu = output_loss[counter]
                        if return_Sigma:
                            Sigma = output_loss[-1]
                        if exp_ev is not None:
                            loss_ev, est_ev = obs_ev_loss(x_ev, mu, exp_ev, xi_ev)
                            loss_value += loss_ev
                            self._train_ev.append(
                                np.concatenate(
                                    ([self._step], (est_ev).detach().to("cpu").numpy())
                                )
                            )
                            if tb_writer is not None:
                                for i in range(est_ev.shape[0]):
                                    tb_writer.add_scalars(
                                        "EV",
                                        {"train_" + str(i + 1): est_ev[i].item()},
                                        self._step,
                                    )
                        if exp_ac is not None:
                            loss_ac, est_ac = obs_ac_loss(
                                x_ac, mu, x_t, K, Sigma, exp_ac, xi_ac
                            )
                            loss_value += loss_ac
                            self._train_ac.append(
                                np.concatenate(
                                    ([self._step], (est_ac).detach().to("cpu").numpy())
                                )
                            )
                            if tb_writer is not None:
                                for i in range(est_ac.shape[0]):
                                    tb_writer.add_scalars(
                                        "AC",
                                        {"train_" + str(i + 1): est_ac[i].item()},
                                        self._step,
                                    )
                        if exp_its is not None:
                            loss_its, est_its = obs_its_loss(
                                S,
                                Sigma,
                                its_state1,
                                its_state2,
                                exp_its,
                                xi_its,
                                epsilon=self.epsilon,
                                mode=self.score_mode,
                            )
                            loss_value += loss_its
                            self._train_its.append(
                                np.concatenate(
                                    ([self._step], (est_its).detach().to("cpu").numpy())
                                )
                            )
                            if tb_writer is not None:
                                for i in range(est_its.shape[0]):
                                    tb_writer.add_scalars(
                                        "ITS",
                                        {"train_" + str(i + 1): est_its[i].item()},
                                        self._step,
                                    )
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            chain(self.ulayer.parameters(), self.slayer.parameters()),
                            CLIP_VALUE,
                        )
                        self.optimizer_u.step()
                        if train_mode == "us":
                            self.optimizer_s.step()

                        if train_score_callback is not None:
                            lval_detached = loss_value.detach()
                            train_score_callback(self._step, lval_detached)
                        self._train_scores.append((self._step, (loss_value).item()))
                        self._train_vampe.append((self._step, (vampe_loss).item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars(
                                "Loss", {"train": loss_value.item()}, self._step
                            )
                            tb_writer.add_scalars(
                                "VAMPE", {"train": vampe_loss.item()}, self._step
                            )
                        if validation_loader is not None:
                            with torch.no_grad():
                                output_loss = vampe_loss_rev(
                                    x_val_0,
                                    x_val_t,
                                    self.ulayer,
                                    self.slayer,
                                    return_mu=return_mu,
                                    return_Sigma=return_Sigma,
                                    return_K=return_K,
                                    return_S=return_S,
                                )
                                vampe_loss = output_loss[0]
                                score_val = -vampe_loss  # vampe loss
                                counter = 1
                                if return_K:
                                    K = output_loss[counter]
                                    counter += 1
                                if return_S:
                                    S = output_loss[counter]
                                    counter += 1
                                if return_mu:
                                    mu = output_loss[counter]
                                if return_Sigma:
                                    Sigma = output_loss[-1]
                                if exp_ev is not None:
                                    loss_ev, est_ev = obs_ev_loss(
                                        x_val_ev, mu, exp_ev, xi_ev
                                    )
                                    score_val += loss_ev
                                    self._validation_ev.append(
                                        np.concatenate(
                                            (
                                                [self._step],
                                                (est_ev).detach().to("cpu").numpy(),
                                            )
                                        )
                                    )
                                    if tb_writer is not None:
                                        for i in range(est_ev.shape[0]):
                                            tb_writer.add_scalars(
                                                "EV",
                                                {
                                                    "valid_"
                                                    + str(i + 1): est_ev[i].item()
                                                },
                                                self._step,
                                            )
                                if exp_ac is not None:
                                    loss_ac, est_ac = obs_ac_loss(
                                        x_val_ac, mu, x_val_t, K, Sigma, exp_ac, xi_ac
                                    )
                                    score_val += loss_ac
                                    self._validation_ac.append(
                                        np.concatenate(
                                            (
                                                [self._step],
                                                (est_ac).detach().to("cpu").numpy(),
                                            )
                                        )
                                    )
                                    if tb_writer is not None:
                                        for i in range(est_ac.shape[0]):
                                            tb_writer.add_scalars(
                                                "AC",
                                                {
                                                    "valid_"
                                                    + str(i + 1): est_ac[i].item()
                                                },
                                                self._step,
                                            )
                                if exp_its is not None:
                                    loss_its, est_its = obs_its_loss(
                                        S,
                                        Sigma,
                                        its_state1,
                                        its_state2,
                                        exp_its,
                                        xi_its,
                                        epsilon=self.epsilon,
                                        mode=self.score_mode,
                                    )
                                    score_val += loss_its
                                    self._validation_its.append(
                                        np.concatenate(
                                            (
                                                [self._step],
                                                (est_its).detach().to("cpu").numpy(),
                                            )
                                        )
                                    )
                                    if tb_writer is not None:
                                        for i in range(est_its.shape[0]):
                                            tb_writer.add_scalars(
                                                "ITS",
                                                {
                                                    "valid_"
                                                    + str(i + 1): est_its[i].item()
                                                },
                                                self._step,
                                            )
                                self._validation_scores.append(
                                    (self._step, score_val.item())
                                )
                                self._validation_vampe.append(
                                    (self._step, vampe_loss.item())
                                )
                                if tb_writer is not None:
                                    tb_writer.add_scalars(
                                        "Loss", {"valid": score_val.item()}, self._step
                                    )
                                    tb_writer.add_scalars(
                                        "VAMPE",
                                        {"valid": vampe_loss.item()},
                                        self._step,
                                    )
                                if validation_score_callback is not None:
                                    validation_score_callback(self._step, score_val)
                        self._step += 1
                if train_mode == "s":
                    with torch.no_grad():
                        output_u = self.ulayer(x_0, x_t, return_mu=return_mu)
                        output_val_u = self.ulayer(
                            x_val_0, x_val_t, return_mu=return_mu
                        )
                        if return_mu:
                            mu = output_u[-1]
                            mu_val = output_val_u[-1]
                        if return_Sigma:
                            Sigma = output_u[4]
                            Sigma_val = output_u[4]
                    for epoch in progress(
                        range(n_epochs),
                        desc="Train s VAMPNet epoch",
                        total=n_epochs,
                        leave=False,
                    ):
                        self.optimizer_s.zero_grad()

                        output_loss = vampe_loss_rev_only_S(
                            *output_u[:5],
                            self.slayer,
                            return_K=return_K,
                            return_S=return_S,
                        )
                        vampe_loss = output_loss[0]
                        loss_value = vampe_loss
                        if return_K:
                            K = output_loss[1]
                        if return_S:
                            S = output_loss[-1]
                        if exp_ac is not None:
                            loss_ac, est_ac = obs_ac_loss(
                                x_ac, mu, x_t, K, Sigma, exp_ac, xi_ac
                            )
                            loss_value += loss_ac
                            self._train_ac.append(
                                np.concatenate(
                                    ([self._step], (est_ac).detach().to("cpu").numpy())
                                )
                            )
                            if tb_writer is not None:
                                for i in range(est_ac.shape[0]):
                                    tb_writer.add_scalars(
                                        "AC",
                                        {"train_" + str(i + 1): est_ac[i].item()},
                                        self._step,
                                    )
                        if exp_its is not None:
                            loss_its, est_its = obs_its_loss(
                                S,
                                Sigma,
                                its_state1,
                                its_state2,
                                exp_its,
                                xi_its,
                                epsilon=self.epsilon,
                                mode=self.score_mode,
                            )
                            loss_value += loss_its
                            self._train_its.append(
                                np.concatenate(
                                    ([self._step], (est_its).detach().to("cpu").numpy())
                                )
                            )
                            if tb_writer is not None:
                                for i in range(est_its.shape[0]):
                                    tb_writer.add_scalars(
                                        "ITS",
                                        {"train_" + str(i + 1): est_its[i].item()},
                                        self._step,
                                    )
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.slayer.parameters(), CLIP_VALUE
                        )
                        self.optimizer_s.step()

                        if train_score_callback is not None:
                            lval_detached = loss_value.detach()
                            train_score_callback(self._step, lval_detached)
                        self._train_scores.append((self._step, (loss_value).item()))
                        self._train_vampe.append((self._step, (vampe_loss).item()))
                        if tb_writer is not None:
                            tb_writer.add_scalars(
                                "Loss", {"train": loss_value.item()}, self._step
                            )
                            tb_writer.add_scalars(
                                "VAMPE", {"train": vampe_loss.item()}, self._step
                            )
                        if validation_loader is not None:
                            with torch.no_grad():
                                output_loss = vampe_loss_rev_only_S(
                                    *output_val_u[:5],
                                    self.slayer,
                                    return_K=return_K,
                                    return_S=return_S,
                                )
                                vampe_loss = output_loss[0]
                                score_val = -vampe_loss
                                if return_K:
                                    K = output_loss[1]
                                if return_S:
                                    S = output_loss[-1]
                                if exp_ac is not None:
                                    loss_ac, est_ac = obs_ac_loss(
                                        x_val_ac,
                                        mu_val,
                                        x_val_t,
                                        K,
                                        Sigma_val,
                                        exp_ac,
                                        xi_ac,
                                    )
                                    score_val += loss_ac
                                    self._validation_ac.append(
                                        np.concatenate(
                                            (
                                                [self._step],
                                                (est_ac).detach().to("cpu").numpy(),
                                            )
                                        )
                                    )
                                    if tb_writer is not None:
                                        for i in range(est_ac.shape[0]):
                                            tb_writer.add_scalars(
                                                "AC",
                                                {
                                                    "valid_"
                                                    + str(i + 1): est_ac[i].item()
                                                },
                                                self._step,
                                            )
                                if exp_its is not None:
                                    loss_its, est_its = obs_its_loss(
                                        S,
                                        Sigma_val,
                                        its_state1,
                                        its_state2,
                                        exp_its,
                                        xi_its,
                                        epsilon=self.epsilon,
                                        mode=self.score_mode,
                                    )
                                    score_val += loss_its
                                    self._validation_its.append(
                                        np.concatenate(
                                            (
                                                [self._step],
                                                (est_its).detach().to("cpu").numpy(),
                                            )
                                        )
                                    )
                                    if tb_writer is not None:
                                        for i in range(est_its.shape[0]):
                                            tb_writer.add_scalars(
                                                "ITS",
                                                {
                                                    "valid_"
                                                    + str(i + 1): est_its[i].item()
                                                },
                                                self._step,
                                            )
                                self._validation_scores.append(
                                    (self._step, score_val.item())
                                )
                                self._validation_vampe.append(
                                    (self._step, vampe_loss.item())
                                )
                                if tb_writer is not None:
                                    tb_writer.add_scalars(
                                        "Loss", {"valid": score_val.item()}, self._step
                                    )
                                    tb_writer.add_scalars(
                                        "VAMPE",
                                        {"valid": vampe_loss.item()},
                                        self._step,
                                    )
                                if validation_score_callback is not None:
                                    validation_score_callback(self._step, score_val)
                        self._step += 1

            return self

    def transform(self, data, instantaneous: bool = True, **kwargs):
        r"""Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        model = self.fetch_model()
        return model.transform(data, **kwargs)

    def fetch_model(self) -> DeepMSMModel:
        r"""Yields the current model."""
        return DeepMSMModel(
            self.lobe,
            self.ulayer,
            self.slayer,
            self.cg_list,
            self.mask,
            dtype=self.dtype,
            device=self.device,
        )

    def set_rev_var(self, data_loader: torch.utils.data.DataLoader, S=False):
        with torch.no_grad():
            chi_t, chi_tau = [], []
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())

        chi_t = torch.cat(chi_t, dim=0)
        chi_tau = torch.cat(chi_tau, dim=0)

        cov_00, cov_0t, cov_tt = covariances(chi_t, chi_tau, remove_mean=False)
        cov_00_inv = (
            sym_inverse(cov_00, epsilon=self.epsilon, mode=self.score_mode)
            .to("cpu")
            .numpy()
        )
        K_vamp = cov_00_inv @ cov_0t.to("cpu").numpy()
        # estimate pi, the stationary distribution vector
        eigv, eigvec = np.linalg.eig(K_vamp.T)
        ind_pi = np.argmin((eigv - 1) ** 2)

        pi_vec = np.real(eigvec[:, ind_pi])
        pi = pi_vec / np.sum(pi_vec, keepdims=True)
        #         print('pi', pi)
        # reverse the consruction of u
        u_optimal = cov_00_inv @ pi
        #         print('u optimal', u_optimal)

        # u_kernel = np.log(np.exp(np.abs(u_optimal))-1) # if softplus
        # for relu
        u_kernel = np.abs(u_optimal)

        with torch.no_grad():
            for param in self.ulayer.parameters():
                param.copy_(torch.Tensor(u_kernel[None, :]))

        if S:
            with torch.no_grad():
                _, _, _, _, Sigma = self.ulayer(chi_t, chi_tau)
                Sigma = Sigma

                sigma_inv = (
                    sym_inverse(Sigma, epsilon=self.epsilon, mode=self.score_mode)
                    .detach()
                    .to("cpu")
                    .numpy()
                )
            # reverse the construction of S
            S_nonrev = K_vamp @ sigma_inv
            S_rev_add = 1 / 2 * (S_nonrev + S_nonrev.T)

            kernel_S = S_rev_add / 2.0
            # for softplus
            # kernel_S = np.log(np.exp(np.abs(kernel_S))-1)
            # for relu
            kernel_S = np.abs(kernel_S)

            with torch.no_grad():
                for param in self.slayer.parameters():
                    param.copy_(torch.Tensor(kernel_S))

    def reset_u_S(self, data_loader: torch.utils.data.DataLoader, reset_opt=False):
        with torch.no_grad():
            chi_t, chi_tau = [], []
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())

        chi_t = torch.cat(chi_t, dim=0)
        chi_tau = torch.cat(chi_tau, dim=0)

        u_kernel = np.ones(self.output_dim)
        K_vamp = np.ones((self.output_dim, self.output_dim)) + np.diag(
            np.ones(self.output_dim)
        )
        K_vamp = K_vamp / np.sum(K_vamp, axis=1, keepdims=True)
        with torch.no_grad():
            for param in self.ulayer.parameters():
                param.copy_(torch.Tensor(u_kernel[None, :]))

        with torch.no_grad():
            _, _, _, _, Sigma = self.ulayer(chi_t, chi_tau)
            Sigma = Sigma

            sigma_inv = (
                sym_inverse(Sigma, epsilon=self.epsilon, mode=self.score_mode)
                .detach()
                .to("cpu")
                .numpy()
            )
        # reverse the construction of S
        S_nonrev = K_vamp @ sigma_inv
        S_rev_add = 1 / 2 * (S_nonrev + S_nonrev.T)

        kernel_S = S_rev_add / 2.0
        # for softplus
        # kernel_S = np.log(np.exp(np.abs(kernel_S))-1)
        # for relu
        kernel_S = np.abs(kernel_S)

        with torch.no_grad():
            for param in self.slayer.parameters():
                param.copy_(torch.Tensor(kernel_S))
        if reset_opt:
            self.optimizer_u = torch.optim.Adam(
                self.ulayer.parameters(), lr=self.learning_rate * 10
            )
            self.optimizer_s = torch.optim.Adam(
                self.slayer.parameters(), lr=self.learning_rate * 100
            )

    def reset_u_S_wo(self):
        u_kernel = np.ones(self.output_dim)

        with torch.no_grad():
            for param in self.ulayer.parameters():
                param.copy_(torch.Tensor(u_kernel[None, :]))
        S_kernel = np.ones((self.output_dim, self.output_dim))
        with torch.no_grad():
            for param in self.slayer.parameters():
                param.copy_(torch.Tensor(S_kernel))

    def reset_opt_u_S(self, lr=1):
        self.optimizer_u = torch.optim.Adam(
            self.ulayer.parameters(), lr=self.learning_rate * 10 * lr
        )
        self.optimizer_s = torch.optim.Adam(
            self.slayer.parameters(), lr=self.learning_rate * 100 * lr
        )

    def reset_opt_all(self, lr=1):
        self.optimizer_lobe = torch.optim.Adam(
            self.lobe.parameters(), lr=self.learning_rate * lr
        )
        self.optimimzer_all = torch.optim.Adam(
            chain(
                self.ulayer.parameters(),
                self.slayer.parameters(),
                self.lobe.parameters(),
            ),
            lr=self.learning_rate * lr,
        )
        self.optimizer_u = torch.optim.Adam(
            self.ulayer.parameters(), lr=self.learning_rate * 10 * lr
        )
        self.optimizer_s = torch.optim.Adam(
            self.slayer.parameters(), lr=self.learning_rate * 100 * lr
        )

    def initialize_cg_layer(
        self, idx: int, data_loader: torch.utils.data.DataLoader, factor: float = 1.0
    ):
        """Initilize the coarse_layer[idx] with the pcca_memberships"""

        assert (
            self.coarse_grain is not None
        ), f"The estimator has no coarse-graining layers"

        assert idx < len(
            self.coarse_grain
        ), f"The chosen idx of the coarse graining layer {idx} does not exist"

        # First estimate the values of u and S before the coarse layer idx
        with torch.no_grad():
            chi_t, chi_tau = [], []
            for batch_0, batch_t in data_loader:
                chi_t.append(self.forward(batch_0).detach())
                chi_tau.append(self.forward(batch_t).detach())
            chi_t = torch.cat(chi_t, dim=0)
            chi_tau = torch.cat(chi_tau, dim=0)

            v, C00, Ctt, C0t, Sigma, u_n = self.ulayer(chi_t, chi_tau, return_u=True)
            _, K_n, S_n = self.slayer(
                v, C00, Ctt, C0t, Sigma, return_S=True, return_K=True
            )

            for cg_id in range(idx):
                _, chi_t, chi_tau, u_n, S_n, K_n = self.cg_list[cg_id].get_cg_uS(
                    chi_t, chi_tau, u_n, S_n, return_chi=True, return_K=True
                )

            T = K_n.to("cpu").numpy().astype("float64")
            # renormalize because of the type casting
            T = T / T.sum(axis=1)[:, None]
            try:
                # use the estimated transition matrix to get the pcca membership
                mem = pcca_memberships(T, self.coarse_grain[idx])
                mem = np.log(mem)
            except ValueError:
                print("PCCA was not successful try different initialization strategy")
                eigvals, eigvecs = np.linalg.eig(T)
                sort_id = np.argsort(eigvals)
                eigvals = eigvals[sort_id]
                eigvecs = eigvecs[:, sort_id]
                size = self.cg_list[idx].M
                mem = eigvecs[:, -size:]
                for j in range(size):
                    ind = np.argwhere(mem[:, j] < 0)
                    mem[ind, j] = mem[ind, j] / np.abs(np.min(mem[:, j]))
                    ind = np.argwhere(mem[:, j] > 0)
                    mem[ind, j] = mem[ind, j] / np.abs(np.max(mem[:, j]))
                mem[:, -1] = mem[:, -1] / size
            # since they will be past through a softmax take log
            initial_values = mem * factor
            # set the parameters of cg_layer[idx] to the estimated values
            self.cg_list[idx].weight.copy_(torch.Tensor(initial_values))
        #             assert np.allclose(mem,self.cg_list[idx].get_softmax().to('cpu').numpy().astype('float64'),rtol=1e-6), 'The estimated values does not match the assigned one'

        return

    def reset_cg(self, idx=0, lr=0.1):
        with torch.no_grad():
            self.cg_list[idx].weight.copy_(
                torch.ones((self.cg_list[idx].N, self.cg_list[idx].M))
            )
        self.cg_opt_list[idx] = torch.optim.Adam(self.cg_list[idx].parameters(), lr=lr)

        return

    def state_dict(self):
        ret = [
            self.lobe.state_dict(),
            self.ulayer.state_dict(),
            self.slayer.state_dict(),
        ]

        if self.coarse_grain is not None:
            for cglayer in self.cg_list:
                ret.append(cglayer.state_dict())
        if len(self.mask.state_dict()) > 0:
            ret.append(self.mask.state_dict())
        return ret

    def load_state_dict(self, dict_lobe, dict_u, dict_S, cg_dicts=None, mask_dict=None):
        self.lobe.load_state_dict(dict_lobe)
        self.ulayer.load_state_dict(dict_u)
        self.slayer.load_state_dict(dict_S)

        if cg_dicts is not None:
            assert len(self.cg_list) == len(
                cg_dicts
            ), "The number of coarse grain layer dictionaries does not match the number of coarse grain layers"
            for i, cglayer in enumerate(self.cg_list):
                cglayer.load_state_dict(cg_dicts[i])
        if mask_dict is not None:
            assert isinstance(self.mask, nn.Module), "The mask layer is not a nn.Module"
            self.mask.load_state_dict(mask_dict)
        return

    def save_params(self, paths: str):
        list_dict = self.state_dict()

        np.savez(
            paths,
            dict_lobe=list_dict[0],
            dict_u=list_dict[1],
            dict_s=list_dict[2],
            *list_dict[3:],
        )

        return print("Saved parameters at: " + paths)

    def load_params(self, paths: str):
        dicts = np.load(paths, allow_pickle=True)

        if len(dicts.keys()) > 3:  # it is expected to be coarse-grain layers
            cg_dicts = []
            for i in range(len(self.cg_list)):
                cg_dicts.append(dicts["arr_" + str(i)].item())
            if len(dicts.keys()) > (3 + i + 1):
                mask_dict = dicts["arr_" + str(i + 1)].item()
            else:
                mask_dict = None
        else:
            cg_dicts = None
            mask_dict = None
        self.load_state_dict(
            dicts["dict_lobe"].item(),
            dicts["dict_u"].item(),
            dicts["dict_s"].item(),
            cg_dicts=cg_dicts,
            mask_dict=mask_dict,
        )

        return
