import warnings
from ..msm import MSMInitializer
from ..util.dataloader import MultimerTrajectoriesDataset, get_symmetrized_data

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import pandas as pd
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

from deeptime.decomposition.deep import VAMPNet
from ..deepmsm.deepmsm import *
from copy import deepcopy

from typing import Optional, Union, Callable, Tuple
from deeptime.decomposition.deep import vampnet_loss, vamp_score
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from sklearn import preprocessing
from scipy.stats import rankdata
from ..decomposition import SymTICA
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

            self.dataset = TrajectoriesDataset.from_numpy(lagtime=self.lag, data=self.feature_trajectories)
            if not self.symmetrize:
                self.dataset_sym = MultimerTrajectoriesDataset.from_numpy(
                    self.lag, self.multimer, self.feature_trajectories
                )
        #            print('The VAMPNETInitializer cannot be saved')
        #                with open(self.filename + 'vampnet_init.pickle', 'wb') as f:
        #                    pickle.dump(self, f)
        else:
            print("The VAMPNETInitializer cannot be loaded")
            self.updating = True
            self.start_analysis()

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
        return [f"{vampnet.n_states}_state_{vampnet.rep}_rep" for vampnet in self._vampnets]

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
            state_probabilities = [self.active_vampnet.transform(traj) for traj in self.dataset.trajectories]
            state_probabilities_concat = np.concatenate(state_probabilities)
            assignments = [stat_prob.argmax(1) for stat_prob in state_probabilities]
            assignments_concat = np.concatenate(assignments)

            self._vampnet_dict[self.active_vampnet_name]["state_probabilities"] = state_probabilities
            self._vampnet_dict[self.active_vampnet_name]["state_probabilities_concat"] = state_probabilities_concat
            self._vampnet_dict[self.active_vampnet_name]["assignments"] = assignments
            self._vampnet_dict[self.active_vampnet_name]["assignments_concat"] = assignments_concat
        self.state_probabilities = self._vampnet_dict[self.active_vampnet_name]["state_probabilities"]
        self.state_probabilities_concat = self._vampnet_dict[self.active_vampnet_name][
            "state_probabilities_concat"
        ]
        self.assignments = self._vampnet_dict[self.active_vampnet_name]["assignments"]
        self.assignments_concat = self._vampnet_dict[self.active_vampnet_name]["assignments_concat"]

    def get_tica_model(self, symmetrize=True):
        print(f"Start TICA with VAMPNET model {self.active_vampnet_name}, lagtime: {self.lag}")
        self.tica = TICA(lagtime=self.lag, observable_transform=self.active_vampnet.fetch_model())
        data_loader = DataLoader(self.dataset, batch_size=20000, shuffle=True)
        for batch_0, batch_t in tqdm(data_loader):
            n_feat_per_sub = batch_0.shape[1] // self.active_vampnet.multimer

            batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)])
            batch_t = torch.concat([torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)])

            self.tica.partial_fit((batch_0.numpy(), batch_t.numpy()))
        self.tica_model = self.tica.fetch_model()

        self._vampnet_dict[self.active_vampnet_name]["tica_model"] = self.tica_model

        if symmetrize:
            self.feature_trajectories = []
            for feature_trajectory in self.dataset.trajectories:
                self.feature_trajectories.extend(get_symmetrized_data([feature_trajectory],
                                                                      self.multimer))
            self.tica_output = [self.tica_model.transform(traj) for traj in self.feature_trajectories]
            self.tica_concatenated = np.concatenate(self.tica_output)
            print("TICA shape:", self.tica_concatenated.shape)
        else:
            self.tica_output = [self.tica_model.transform(traj) for traj in self.dataset.trajectories]
            self.tica_concatenated = np.concatenate(self.tica_output)
            print("TICA shape:", self.tica_concatenated.shape)
        self.transformer = self.tica


class VAMPNETInitializer_Multimer(VAMPNETInitializer):
    def select_vampnet(self, index, update=False):
        """Selects the VAMPNET model to be used for further analysis.
        When the VAMPNET model is not yet in the dictionary, it is calculated and stored.
        After selection, the state probabilities and assignments are stored in the class.
        """
        self.active_vampnet = self.vampnets[index]
        self.transformer = self.active_vampnet
        self.active_vampnet_name = self.state_list[index]
        print(
            f"The activated VAMPNET # index: {index} # states: {self.active_vampnet.n_states} # rep: {self.active_vampnet.rep}"
        )

        if not self.vampnet_dict[self.active_vampnet_name] or update:
            state_probabilities = [self.active_vampnet.transform(traj) for traj in self.dataset.trajectories]
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
                        * (self.active_vampnet.n_states ** np.arange(self.active_vampnet.multimer)),
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
                cluster_rank_dtrajs.append(cluster_rank_concat[curr_ind : curr_ind + sub_dtrajs.shape[0]])
                curr_ind += sub_dtrajs.shape[0]

            self._vampnet_dict[self.active_vampnet_name]["state_probabilities"] = state_probabilities
            self._vampnet_dict[self.active_vampnet_name]["state_probabilities_concat"] = state_probabilities_concat
            self._vampnet_dict[self.active_vampnet_name]["assignments"] = assignments
            self._vampnet_dict[self.active_vampnet_name]["assignments_concat"] = assignments_concat
            self._vampnet_dict[self.active_vampnet_name]["cluster_degen_dtrajs"] = cluster_degen_dtrajs
            self._vampnet_dict[self.active_vampnet_name]["cluster_degen_concat"] = cluster_degen_concat
            self._vampnet_dict[self.active_vampnet_name]["cluster_rank_dtrajs"] = cluster_rank_dtrajs
            self._vampnet_dict[self.active_vampnet_name]["cluster_rank_concat"] = cluster_rank_concat
            self._vampnet_dict[self.active_vampnet_name]["stat_rank_mapping"] = self.get_assignment_rank_mapping(
                assignments_concat, cluster_rank_concat
            )

        self.state_probabilities = self._vampnet_dict[self.active_vampnet_name]["state_probabilities"]
        self.state_probabilities_concat = self._vampnet_dict[self.active_vampnet_name][
            "state_probabilities_concat"
        ]
        self.assignments = self._vampnet_dict[self.active_vampnet_name]["assignments"]
        self.assignments_concat = self._vampnet_dict[self.active_vampnet_name]["assignments_concat"]
        self.cluster_degen_dtrajs = self._vampnet_dict[self.active_vampnet_name]["cluster_degen_dtrajs"]
        self.cluster_degen_concat = self._vampnet_dict[self.active_vampnet_name]["cluster_degen_concat"]
        self.cluster_rank_dtrajs = self._vampnet_dict[self.active_vampnet_name]["cluster_rank_dtrajs"]
        self.cluster_rank_concat = self._vampnet_dict[self.active_vampnet_name]["cluster_rank_concat"]
        self.stat_rank_mapping = self._vampnet_dict[self.active_vampnet_name]["stat_rank_mapping"]

    @staticmethod
    def get_assignment_rank_mapping(assignment, cluster_rank):
        stat_rank_mapping = {}
        for i in range(cluster_rank.max() + 1):
            stat_rank_mapping[i] = assignment[np.where(cluster_rank == i)[0][0]]
        return stat_rank_mapping

    def generate_state_dataframe(self, tica_output=None):
        start_time = self.start * self.dt * 1000
        self.state_df = self.md_dataframe.dataframe[self.md_dataframe.dataframe.traj_time >= start_time].reset_index(drop=True).copy()
        if tica_output is not None:
            plotly_tica_concatenated = np.concatenate(tica_output[::])
            self.state_df["tic_1"] = plotly_tica_concatenated[:, 0]
            self.state_df["tic_2"] = plotly_tica_concatenated[:, 1]

        for ind, vampnet in enumerate(self.vampnets):
            self.select_vampnet(ind)
            self.state_df = pd.concat(
                [
                    self.state_df,
                    pd.DataFrame(
                        self.cluster_rank_concat,
                        columns=[f"n_states_{self.active_vampnet.n_states}_rep_{self.active_vampnet.rep}"],
                    ),
                ],
                axis=1,
            )
            self.state_df = pd.concat(
                [
                    self.state_df,
                    pd.DataFrame(
                        self.assignments_concat,
                        columns=[
                            f"n_states_{self.active_vampnet.n_states}_sub_{subunit}_rep_{self.active_vampnet.rep}"
                            for subunit in range(self.multimer)
                        ],
                    ),
                ],
                axis=1,
            )
        self.select_vampnet(0)

    def get_sym_tica_model(self, overwrite_tica=True):
        print(f"Start SymTICA with VAMPNET model {self.active_vampnet_name}, lagtime: {self.lag}")
        self.sym_tica = SymTICA(
            symmetry_fold=self.active_vampnet.multimer,
            lagtime=self.lag,
            observable_transform=self.active_vampnet.fetch_model(),
        )
        data_loader = DataLoader(self.dataset, batch_size=20000, shuffle=True)
        for batch_0, batch_t in tqdm(data_loader):
            n_feat_per_sub = batch_0.shape[1] // self.active_vampnet.multimer

            batch_0 = torch.concat([torch.roll(batch_0, n_feat_per_sub * i, 1) for i in range(self.multimer)])
            batch_t = torch.concat([torch.roll(batch_t, n_feat_per_sub * i, 1) for i in range(self.multimer)])

            self.sym_tica.partial_fit((batch_0.numpy(), batch_t.numpy()))
        self.sym_tica_model = self.sym_tica.fetch_model()

        self._vampnet_dict[self.active_vampnet_name]["sym_tica_model"] = self.sym_tica_model

        self.sym_tica_output = [self.sym_tica_model.transform(traj) for traj in self.dataset.trajectories]
        self.sym_tica_concatenated = np.concatenate(self.sym_tica_output)
        print("TICA shape:", self.sym_tica_concatenated.shape)
        self.transformer = self.sym_tica
        if overwrite_tica:
            self.tica_output = self.sym_tica_output
            self.tica_concatenated = self.sym_tica_concatenated
