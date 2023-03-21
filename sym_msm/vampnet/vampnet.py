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
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
from deeptime.decomposition import VAMP, TICA
from torch.utils.data import DataLoader

from deeptime.decomposition.deep import VAMPNet
from ..deepmsm.deepmsm import *
from copy import deepcopy

from typing import Optional, Union, Callable, Tuple
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from sklearn import preprocessing
from scipy.stats import rankdata
from ..tica.sym_tica import SymTICA
from .score import vamp_score_sym, vamp_score_rev


class VAMPNETInitializer(MSMInitializer):
    prefix = "vampnet"

    def start_analysis(self):
        self._vampnets = []
        self._vampnet_dict = {}
        os.makedirs(self.filename, exist_ok=True)

        if (not os.path.isfile(self.filename + "vampnet.pyemma")) or self.updating:
            print("Start new VAMPNET analysis")
            if self.in_memory:
                if not self.data_collected:
                    self.gather_feature_matrix()
            else:
                print("Partial fitting is not supported in VAMPNET")
                if not self.data_collected:
                    self.gather_feature_matrix()

            # self.dataset = MultimerTrajectoriesDataset.from_numpy(
            #    self.lag, self.multimer, self.feature_trajectories)

            self.dataset = TrajectoriesDataset.from_numpy(
                lagtime=self.lag, data=self.feature_trajectories
            )
            if not self.symmetrize:
                self.dataset_sym = MultimerTrajectoriesDataset.from_numpy(
                    self.lag, self.multimer, self.feature_trajectories
                )
            if self.dumping:
                self.dump_feature_trajectories()

        #            print('The VAMPNETInitializer cannot be saved')
        #                with open(self.filename + 'vampnet_init.pickle', 'wb') as f:
        #                    pickle.dump(self, f)
        else:
            print("Load old VAMPNET results")

    #            self = pickle.load(open(self.filename + 'vampnet_init.pickle', 'rb'))

    @property
    def vampnets(self):
        return self._vampnets

    @vampnets.setter
    def vampnets(self, value):
        self._vampnets = value
        self.select_vampnet(0)

    @property
    def state_list(self):
        return [
            f"{vampnet.n_states}_state_{vampnet.rep}_rep" for vampnet in self._vampnets
        ]

    @property
    def vampnet_dict(self):
        if not self._vampnet_dict:
            self._vampnet_dict = {key: {} for key in self.state_list}
        return self._vampnet_dict

    def select_vampnet(self, index, update=False):
        self.active_vampnet = self.vampnets[index]
        self.active_vampnet_name = self.state_list[index]
        print("The activated VAMPNET # states:", self.active_vampnet.n_states)
        print("The activated VAMPNET # rep:", self.active_vampnet.rep)

        if not self.vampnet_dict[self.active_vampnet_name] or update:
            state_probabilities = [
                self.active_vampnet.transform(traj)
                for traj in self.dataset.trajectories
            ]
            state_probabilities_concat = np.concatenate(state_probabilities)
            assignments = [stat_prob.argmax(1) for stat_prob in state_probabilities]
            assignments_concat = np.concatenate(assignments)

            self._vampnet_dict[self.active_vampnet_name][
                "state_probabilities"
            ] = state_probabilities
            self._vampnet_dict[self.active_vampnet_name][
                "state_probabilities_concat"
            ] = state_probabilities_concat
            self._vampnet_dict[self.active_vampnet_name]["assignments"] = assignments
            self._vampnet_dict[self.active_vampnet_name][
                "assignments_concat"
            ] = assignments_concat
        self.state_probabilities = self._vampnet_dict[self.active_vampnet_name][
            "state_probabilities"
        ]
        self.state_probabilities_concat = self._vampnet_dict[self.active_vampnet_name][
            "state_probabilities_concat"
        ]
        self.assignments = self._vampnet_dict[self.active_vampnet_name]["assignments"]
        self.assignments_concat = self._vampnet_dict[self.active_vampnet_name][
            "assignments_concat"
        ]

    def get_tica_model(self):
        print(
            f"Start TICA with VAMPNET model {self.active_vampnet_name}, lagtime: {self.lag}"
        )
        self.tica = TICA(
            lagtime=self.lag, observable_transform=self.active_vampnet.fetch_model()
        )
        data_loader = DataLoader(self.dataset, batch_size=20000, shuffle=True)
        for batch_0, batch_t in tqdm(data_loader):
            n_feat_per_sub = batch_0.shape[1] // self.active_vampnet.multimer

            batch_0 = torch.concat(
                [
                    torch.roll(batch_0, n_feat_per_sub * i, 1)
                    for i in range(self.multimer)
                ]
            )
            batch_t = torch.concat(
                [
                    torch.roll(batch_t, n_feat_per_sub * i, 1)
                    for i in range(self.multimer)
                ]
            )

            self.tica.partial_fit((batch_0.numpy(), batch_t.numpy()))
        self.tica_model = self.tica.fetch_model()

        self._vampnet_dict[self.active_vampnet_name]["tica_model"] = self.tica_model

        self.tica_output = [
            self.tica_model.transform(traj) for traj in self.dataset.trajectories
        ]
        self.tica_concatenated = np.concatenate(self.tica_output)
        print("TICA shape:", self.tica_concatenated.shape)


class VAMPNETInitializer_Multimer(VAMPNETInitializer):
    def select_vampnet(self, index, update=False):
        self.active_vampnet = self.vampnets[index]
        self.active_vampnet_name = self.state_list[index]
        print(
            f"The activated VAMPNET # index: {index} # states: {self.active_vampnet.n_states} # rep: {self.active_vampnet.rep}"
        )

        if not self.vampnet_dict[self.active_vampnet_name] or update:
            state_probabilities = [
                self.active_vampnet.transform(traj)
                for traj in self.dataset.trajectories
            ]
            state_probabilities_concat = np.concatenate(state_probabilities)
            assignments = [
                stat_prob.reshape(
                    stat_prob.shape[0],
                    self.active_vampnet.multimer,
                    self.active_vampnet.n_states,
                ).argmax(2)
                for stat_prob in state_probabilities
            ]
            assignments_concat = np.concatenate(assignments)
            cluster_degen_dtrajs = []
            for sub_dtrajs in assignments:
                #    degenerated_traj = np.apply_along_axis(convert_state_to_degenerated, axis=1, arr=sub_dtrajs)
                sorted_sub_dtrajs = np.sort(sub_dtrajs, axis=1)[:, ::-1]
                cluster_degen_dtrajs.append(
                    np.sum(
                        sorted_sub_dtrajs
                        * (
                            self.active_vampnet.n_states
                            ** np.arange(self.active_vampnet.multimer)
                        ),
                        axis=1,
                    )
                )
            cluster_degen_concat = np.concatenate(cluster_degen_dtrajs)
            cluster_rank_concat = rankdata(cluster_degen_concat, method="dense") - 1
            print("# of cluster", cluster_rank_concat.max() + 1)
            self.n_clusters = cluster_rank_concat.max() + 1
            cluster_rank_dtrajs = []
            curr_ind = 0
            for sub_dtrajs in assignments:
                cluster_rank_dtrajs.append(
                    cluster_rank_concat[curr_ind : curr_ind + sub_dtrajs.shape[0]]
                )
                curr_ind += sub_dtrajs.shape[0]

            self._vampnet_dict[self.active_vampnet_name][
                "state_probabilities"
            ] = state_probabilities
            self._vampnet_dict[self.active_vampnet_name][
                "state_probabilities_concat"
            ] = state_probabilities_concat
            self._vampnet_dict[self.active_vampnet_name]["assignments"] = assignments
            self._vampnet_dict[self.active_vampnet_name][
                "assignments_concat"
            ] = assignments_concat
            self._vampnet_dict[self.active_vampnet_name][
                "cluster_degen_dtrajs"
            ] = cluster_degen_dtrajs
            self._vampnet_dict[self.active_vampnet_name][
                "cluster_degen_concat"
            ] = cluster_degen_concat
            self._vampnet_dict[self.active_vampnet_name][
                "cluster_rank_dtrajs"
            ] = cluster_rank_dtrajs
            self._vampnet_dict[self.active_vampnet_name][
                "cluster_rank_concat"
            ] = cluster_rank_concat
            self._vampnet_dict[self.active_vampnet_name][
                "stat_rank_mapping"
            ] = self.get_assignment_rank_mapping(
                assignments_concat, cluster_rank_concat
            )

        self.state_probabilities = self._vampnet_dict[self.active_vampnet_name][
            "state_probabilities"
        ]
        self.state_probabilities_concat = self._vampnet_dict[self.active_vampnet_name][
            "state_probabilities_concat"
        ]
        self.assignments = self._vampnet_dict[self.active_vampnet_name]["assignments"]
        self.assignments_concat = self._vampnet_dict[self.active_vampnet_name][
            "assignments_concat"
        ]
        self.cluster_degen_dtrajs = self._vampnet_dict[self.active_vampnet_name][
            "cluster_degen_dtrajs"
        ]
        self.cluster_degen_concat = self._vampnet_dict[self.active_vampnet_name][
            "cluster_degen_concat"
        ]
        self.cluster_rank_dtrajs = self._vampnet_dict[self.active_vampnet_name][
            "cluster_rank_dtrajs"
        ]
        self.cluster_rank_concat = self._vampnet_dict[self.active_vampnet_name][
            "cluster_rank_concat"
        ]
        self.stat_rank_mapping = self._vampnet_dict[self.active_vampnet_name][
            "stat_rank_mapping"
        ]

    @staticmethod
    def get_assignment_rank_mapping(assignment, cluster_rank):
        stat_rank_mapping = {}
        for i in range(cluster_rank.max() + 1):
            stat_rank_mapping[i] = assignment[np.where(cluster_rank == i)[0][0]]
        return stat_rank_mapping

    def get_tica_model(self):
        print(
            f"Start SymTICA with VAMPNET model {self.active_vampnet_name}, lagtime: {self.lag}"
        )
        self.tica = SymTICA(
            symmetry_fold=self.active_vampnet.multimer,
            lagtime=self.lag,
            observable_transform=self.active_vampnet.fetch_model(),
        )
        data_loader = DataLoader(self.dataset, batch_size=20000, shuffle=True)
        for batch_0, batch_t in tqdm(data_loader):
            n_feat_per_sub = batch_0.shape[1] // self.active_vampnet.multimer

            batch_0 = torch.concat(
                [
                    torch.roll(batch_0, n_feat_per_sub * i, 1)
                    for i in range(self.multimer)
                ]
            )
            batch_t = torch.concat(
                [
                    torch.roll(batch_t, n_feat_per_sub * i, 1)
                    for i in range(self.multimer)
                ]
            )

            self.tica.partial_fit((batch_0.numpy(), batch_t.numpy()))
        self.tica_model = self.tica.fetch_model()

        self._vampnet_dict[self.active_vampnet_name]["tica_model"] = self.tica_model

        self.tica_output = [
            self.tica_model.transform(traj) for traj in self.dataset.trajectories
        ]
        self.tica_concatenated = np.concatenate(self.tica_output)
        print("TICA shape:", self.tica_concatenated.shape)


class MultimerNet(nn.Module):
    def __init__(self, data_shape, multimer, n_states):
        super().__init__()
        self.data_shape = data_shape
        self.multimer = multimer
        self.n_states = n_states

        self.n_feat_per_sub = self.data_shape // self.multimer
        self._construct_architecture()

    def _construct_architecture(self):
        self.batchnorm1d = nn.BatchNorm1d(self.n_feat_per_sub)

        # Fully connected layers into monomer part
        self.fc1 = nn.Linear(self.n_feat_per_sub, 200)
        self.elu1 = nn.ELU()

        self.fc2 = nn.Linear(200, 100)
        self.elu2 = nn.ELU()

        self.fc3 = nn.Linear(100, 50)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(50, 20)
        self.elu4 = nn.ELU()

        self.fc5 = nn.Linear(20, self.n_states)
        self.softmax = nn.Softmax(dim=1)

        #        self.fc6 = nn.Linear(20, 2)
        #        self.elu6 = nn.ELU()

        #        self.fc7 = nn.Linear(2, self.n_states)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    # x represents our data
    def forward(self, x):
        #       x = self.batchnorm1d(x)

        batch_size = x.shape[0]

        n_feat_per_sub = int(self.data_shape / self.multimer)
        x_splits = x.reshape(batch_size, self.multimer, self.n_feat_per_sub)
        output = []

        x_stack = torch.permute(x_splits, (1, 0, 2)).reshape(
            batch_size * self.multimer, self.n_feat_per_sub
        )

        x_stack = self.batchnorm1d(x_stack)
        x_stack = self.fc1(x_stack)
        x_stack = self.elu1(x_stack)
        x_stack = self.dropout1(x_stack)
        x_stack = self.fc2(x_stack)
        x_stack = self.elu2(x_stack)
        x_stack = self.dropout2(x_stack)
        x_stack = self.fc3(x_stack)
        x_stack = self.elu3(x_stack)
        x_stack = self.fc4(x_stack)
        x_stack = self.elu4(x_stack)
        x_stack = self.fc5(x_stack)
        #        x_stack = self.fc6(x_stack)
        #        x_stack = self.elu6(x_stack)
        #        x_stack = self.fc7(x_stack)
        x_stack = self.softmax(x_stack)

        x_splits = (
            x_stack.reshape(self.multimer, batch_size, self.n_states)
            .permute(1, 0, 2)
            .reshape(batch_size, self.n_states * self.multimer)
        )
        return x_splits


class VAMPNet_Multimer(VAMPNet):
    def __init__(
        self,
        multimer: int,
        n_states: int,
        lobe: nn.Module,
        sym: bool = False,
        rep: int = 0,
        lobe_timelagged: Optional[nn.Module] = None,
        device=None,
        optimizer: Union[str, Callable] = "Adam",
        learning_rate: float = 5e-4,
        score_method: str = "VAMP2",
        score_mode: str = "regularize",
        epsilon: float = 1e-6,
        dtype=np.float32,
        trained=False,
    ):
        super().__init__(
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

        self.multimer = multimer
        self.n_states = n_states
        self.rep = rep
        self.trained = trained

        # if the output is symmetric over subunits,
        # set sym to True
        self.sym = sym

        if self.sym:
            self._train_scores_full = []
            self._validation_scores_full = []
            self._train_scores_deg = []
            self._validation_scores_deg = []

    def prepare_partial_fit(self):
        self.trained = True
        if self.dtype == np.float32:
            self._lobe = self._lobe.float()
            self._lobe_timelagged = self._lobe_timelagged.float()
        elif self.dtype == np.float64:
            self._lobe = self._lobe.double()
            self._lobe_timelagged = self._lobe_timelagged.double()

        self.lobe.train()
        self.lobe_timelagged.train()

    def partial_fit(
        self,
        data,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
    ):
        self.prepare_partial_fit()

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

        self.optimizer.zero_grad()
        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        loss_value = vampnet_loss(
            x_0,
            x_t,
            method=self.score_method,
            epsilon=self.epsilon,
            mode=self.score_mode,
        )

        loss_value.backward()
        self.optimizer.step()

        if self.sym:
            score_value_full = -vampnet_loss(
                x_0,
                x_t,
                method=self.score_method,
                epsilon=self.epsilon,
                mode=self.score_mode,
            )
            score_value_deg = vamp_score_sym(
                x_0,
                x_t,
                symmetry_fold=self.multimer,
                method=self.score_method,
                epsilon=self.epsilon,
                mode=self.score_mode,
            )
        else:
            loss_value_full = None
            loss_value_deg = None
        self.append_training_score(
            self._step, -loss_value, score_value_full, score_value_deg
        )

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        if tb_writer is not None:
            tb_writer.add_scalars("Loss", {"train": loss_value.item()}, self._step)
            tb_writer.add_scalars("VAMPE", {"train": -loss_value.item()}, self._step)
        self._step += 1

        return self

    def append_training_score(self, step, score, score_full=None, score_deg=None):
        self._train_scores.append((step, score.item()))
        if self.sym:
            self._train_scores_full.append((step, score_full.item()))
            self._train_scores_deg.append((step, score_deg.item()))

    def append_validation_score(self, step, score, score_full=None, score_deg=None):
        self._validation_scores.append((step, score.item()))
        if self.sym:
            self._validation_scores_full.append((step, score_full.item()))
            self._validation_scores_deg.append((step, score_deg.item()))

    @property
    def train_scores_full(self) -> np.ndarray:
        r"""The collected train scores for full mat. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores_full)

    @property
    def train_scores_deg(self) -> np.ndarray:
        r"""The collected train scores for degenerated mat. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores_deg)

    @property
    def validation_scores_full(self) -> np.ndarray:
        r"""The collected validation scores for full mat. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores_full)

    @property
    def validation_scores_deg(self) -> np.ndarray:
        r"""The collected validation scores for degenerated mat. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores_deg)

    def validate(self, validation_data: Tuple[torch.Tensor]):
        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])
                # augmenting validation set by permutation
                val_aug = torch.concat(
                    [
                        torch.roll(val, self.n_states * i, 1)
                        for i in range(self.multimer)
                    ]
                )
                val_t_aug = torch.concat(
                    [
                        torch.roll(val_t, self.n_states * i, 1)
                        for i in range(self.multimer)
                    ]
                )
                score_value = vamp_score(
                    val_aug,
                    val_t_aug,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon,
                )
                if self.sym:
                    score_value_full = vamp_score(
                        val,
                        val_t,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon,
                    )
                    score_value_deg = vamp_score_sym(
                        val_aug,
                        val_t,
                        symmetry_fold=self.multimer,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon,
                    )
                    return score_value, score_value_full, score_value_deg
                return score_value

    def transform(self, data, **kwargs):
        r"""Transforms data with the encapsulated model.

        Parameters
        ----------
        data : array_like
            Input data
        **kwargs
            Optional arguments.

        Returns
        -------
        output : array_like
            Transformed data.
        """
        if not self.trained:
            warnings.warn("VAMPNet not trained yet. Please call fit first.")
        model = self.fetch_model()
        if model is None:
            raise ValueError(
                "This estimator contains no model yet, fit should be called first."
            )
        return model.transform(data, **kwargs)

    def fit(
        self,
        data_loader: "torch.utils.data.DataLoader",
        n_epochs=1,
        validation_loader=None,
        train_score_callback: Callable[[int, "torch.Tensor"], None] = None,
        validation_score_callback: Callable[[int, "torch.Tensor"], None] = None,
        progress=None,
        early_stopping_patience=None,
        early_stopping_threshold=0.0,
        **kwargs,
    ):
        r"""Fits a VampNet on data with early stopping.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        progress : context manager, optional, default=None
            Progress bar (eg tqdm), defaults to None.
        early_stopping_patience : int, optional, default=None
            If given, the training will stop after the given number of epochs without improvement of the
            validation score. If None, no early stopping is performed.
        early_stopping_threshold : float, optional, default=0.0
            The training will stop if the validation score does not improve by at least the given
            threshold.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : VAMPNet
            Reference to self.
        """
        print("VAMPNet training with {} epochs".format(n_epochs))
        print("Early stopping patience: {}".format(early_stopping_patience))
        print("Early stopping threshold: {}".format(early_stopping_threshold))

        from deeptime.util.platform import handle_progress_bar

        progress = handle_progress_bar(progress)
        self._step = 0

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_validation_score = -np.inf
        self.best_validation_score_epoch = 0
        self.best_model = None

        # and train
        with disable_TF32():
            for _ in progress(
                range(n_epochs), desc="VAMPNet epoch", total=n_epochs, leave=False
            ):
                for batch_0, batch_t in data_loader:
                    self.partial_fit(
                        (
                            batch_0.to(device=self.device),
                            batch_t.to(device=self.device),
                        ),
                        train_score_callback=train_score_callback,
                    )
                if validation_loader is not None:
                    with torch.no_grad():
                        scores = []
                        if self.sym:
                            scores_full = []
                            scores_deg = []
                        for val_batch in validation_loader:
                            if self.sym:
                                (
                                    val_score,
                                    val_score_full,
                                    val_score_deg,
                                ) = self.validate(
                                    (
                                        val_batch[0].to(device=self.device),
                                        val_batch[1].to(device=self.device),
                                    )
                                )
                                scores.append(val_score)
                                scores_full.append(val_score_full)
                                scores_deg.append(val_score_deg)
                            else:
                                val_score = self.validate(
                                    (
                                        val_batch[0].to(device=self.device),
                                        val_batch[1].to(device=self.device),
                                    )
                                )
                                scores.append(val_score)
                        mean_score = torch.mean(torch.stack(scores))
                        if self.sym:
                            mean_score_full = torch.mean(torch.stack(scores_full))
                            mean_score_deg = torch.mean(torch.stack(scores_deg))
                        else:
                            mean_score_full = None
                            mean_score_deg = None
                        self.append_validation_score(
                            self._step, mean_score, mean_score_full, mean_score_deg
                        )
                        if validation_score_callback is not None:
                            validation_score_callback(self._step, mean_score)

                        if (
                            mean_score
                            > self.best_validation_score + early_stopping_threshold
                        ):
                            self.best_validation_score = mean_score
                            self.best_validation_score_epoch = self._step
                            self.best_model = self.fetch_model()
                        if early_stopping_patience is not None:
                            if (
                                self._step - self.best_validation_score_epoch
                                > early_stopping_patience
                            ):
                                self._lobe = self.best_model._lobe
                                self._lobe_timelagged = self.best_model._lobe_timelagged
                                self._step = self.best_validation_score_epoch
                                print(
                                    "Early stopping after {} epochs without improvement.".format(
                                        early_stopping_patience
                                    )
                                )
                                print(
                                    "Best validation score: {}".format(
                                        self.best_validation_score
                                    )
                                )
                                return self
        return self

    def save(self, folder, n_epoch, rep=None):
        if rep is None:
            rep = 0

        pickle.dump(
            self,
            open(
                f"{folder}/{self.__class__.__name__}/epoch_{n_epoch}_state_{self.n_states}_rep_{rep}.lobe",
                "wb",
            ),
        )


class VAMPNet_Multimer_AUG(VAMPNet_Multimer):
    def partial_fit(
        self,
        data,
        train_score_callback: Callable[[int, torch.Tensor], None] = None,
        tb_writer=None,
    ):
        self.prepare_partial_fit()

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

        self.optimizer.zero_grad()

        n_feat_per_sub = batch_0.shape[1] // self.multimer

        # augmenting training set by permutation
        batch_0 = torch.concat(
            [torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)]
        )
        batch_t = torch.concat(
            [torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)]
        )

        x_0 = self.lobe(batch_0)
        x_t = self.lobe_timelagged(batch_t)

        loss_value = vampnet_loss(
            x_0,
            x_t,
            method=self.score_method,
            epsilon=self.epsilon,
            mode=self.score_mode,
        )

        loss_value.backward()
        self.optimizer.step()

        if self.sym:
            score_value_full = -vampnet_loss(
                x_0,
                x_t,
                method=self.score_method,
                epsilon=self.epsilon,
                mode=self.score_mode,
            )
            score_value_deg = vamp_score_sym(
                x_0,
                x_t,
                symmetry_fold=self.multimer,
                method=self.score_method,
                epsilon=self.epsilon,
                mode=self.score_mode,
            )
        else:
            loss_value_full = None
            loss_value_deg = None
        self.append_training_score(
            self._step, -loss_value, score_value_full, score_value_deg
        )

        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        if tb_writer is not None:
            tb_writer.add_scalars("Loss", {"train": loss_value.item()}, self._step)
            tb_writer.add_scalars("VAMPE", {"train": -loss_value.item()}, self._step)
        self._step += 1

        return self

    def validate(self, validation_data: Tuple[torch.Tensor]):
        with disable_TF32():
            self.lobe.eval()
            self.lobe_timelagged.eval()

            with torch.no_grad():
                val = self.lobe(validation_data[0])
                val_t = self.lobe_timelagged(validation_data[1])
                # augmenting validation set by permutation
                val_aug = torch.concat(
                    [
                        torch.roll(val, self.n_states * i, 1)
                        for i in range(self.multimer)
                    ]
                )
                val_t_aug = torch.concat(
                    [
                        torch.roll(val_t, self.n_states * i, 1)
                        for i in range(self.multimer)
                    ]
                )
                score_value = vamp_score(
                    val_aug,
                    val_t_aug,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon,
                )
                if self.sym:
                    score_value_full = vamp_score(
                        val_aug,
                        val_t_aug,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon,
                    )
                    score_value_deg = vamp_score_sym(
                        val_aug,
                        val_t_aug,
                        symmetry_fold=self.multimer,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon,
                    )
                    return score_value, score_value_full, score_value_deg

                return score_value