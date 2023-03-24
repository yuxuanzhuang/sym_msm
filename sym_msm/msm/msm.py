import os
import numpy as np

import warnings

# ignore pandas future warning
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt

# import pyemma
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
from deeptime.plots import plot_implied_timescales, plot_ck_test
from deeptime.util.validation import implied_timescales, ImpliedTimescales
from ENPMDA import MDDataFrame
from joblib import Parallel, delayed
import pickle
from datetime import datetime
from copy import deepcopy
from scipy.stats import rankdata

from typing import List, Optional

import gc
import itertools
import MDAnalysis as mda
from MDAnalysis.analysis import align
from tqdm import tqdm
from ..util.utils import tqdm_joblib
from ..util.score_cv import score_cv
from ..util.dataloader import MultimerTrajectoriesDataset, get_symmetrized_data


class MSMInitializer:
    prefix = "msm"

    def __init__(
        self,
        md_dataframe: MDDataFrame,
        lag: int,
        multimer: int,
        start: int = 0,
        end: int = -1,
        symmetrize: bool = True,
        updating: bool = True,
        system_exclusion: List = [],
        prefix: Optional[str] = None,
        in_memory: bool = True,
    ):
        """MSM initializer

        Wrapper of all functionality to build a MSM

        Parameters
        ----------
        md_dataframe: MDDataframe
            MDDataframe object
        lag: int
            lag time in frames
        multimer: int
            number of subunits in the multimer
        start: int
            starting frame
        end: int
            ending frame
        symmetrize: bool
            whether to symmetrize the data
        updating: bool
            whether to update the MSM
        system_exclusion: list
            list of systems to exclude
        prefix: str
            prefix of the MSM
        in_memory: bool
            whether to load the data into memory
        """

        self.md_dataframe = deepcopy(md_dataframe)

        # lag for # of frames
        self.lag = lag
        self.start = start
        self.end = end
        self.multimer = multimer
        self.symmetrize = symmetrize
        self.updating = updating

        self.system_exclusion = system_exclusion

        if prefix != None:
            self.prefix = prefix
        self.in_memory = in_memory
        self.data_collected = False

        self.ck_test = {}
        self.rerun_msm = False

        self.md_dataframe.dataframe = self.md_dataframe.dataframe[
            self.md_dataframe.dataframe.system.isin(self.system_exclusion) == False
        ].reset_index(drop=True)
        system_array = self.md_dataframe.dataframe.system.to_numpy()
        self.n_trajectories = len(np.unique(system_array))

        def fill_missing_values(system_array):
            diff_arr = (np.diff(system_array, prepend=0) != 0) & (np.diff(system_array, prepend=0) != 1)
            if all(diff_arr == False):
                return system_array
            start_index_update = np.arange(system_array.shape[0])[diff_arr][0]
            system_array[start_index_update:] = system_array[start_index_update:] - 1
            return fill_missing_values(system_array)

        system_array = fill_missing_values(system_array)
        self.md_dataframe.dataframe.system = system_array
        self.md_data = self.md_dataframe.dataframe
        # dt: ns
        self.dt = (self.md_data.traj_time[1] - self.md_data.traj_time[0]) / 1000
        print(f"lag time is {self.lag * self.dt} ns")
        print(f"start time is {self.start * self.dt} ns")
        if self.end != -1:
            print(f"end time is {self.end * self.dt} ns")

        self.feature_input_list = []
        self.feature_input_info_list = []
        self.feature_input_indice_list = []
        self.feature_type_list = []

        os.makedirs(self.filename, exist_ok=True)

    def add_feature(self, feature_selected, excluded_indices=[], feat_type="subunit"):
        if feature_selected not in self.md_dataframe.analysis_list:
            raise ValueError(
                f"feature selection {feature_selected} not available\n"
                f"Available features are {self.md_dataframe.analysis_list}"
            )
        self.feature_input_list.append(feature_selected)

        feature_info = np.load(
            self.md_dataframe.analysis_results.filename + feature_selected + "_feature_info.npy"
        )

        self.feature_input_info_list.append(np.delete(feature_info, excluded_indices))
        self.feature_input_indice_list.append(np.delete(np.arange(len(feature_info)), excluded_indices))
        self.feature_type_list.append(feat_type)
        print(
            f"added feature selection {feature_selected} type: {feat_type}, # of features: {len(self.feature_input_info_list[-1])}"
        )

    def gather_feature_matrix(self):
        """load feature matrix into memory"""
        if self.feature_input_list == []:
            raise ValueError("No feature selected yet, please use add_feature() first")
        self.feature_trajectories = []
        feature_df = self.md_dataframe.get_feature(self.feature_input_list, in_memory=False)
        for system, row in tqdm(feature_df.iterrows(), total=feature_df.shape[0]):
            feature_trajectory = []
            for feat_loc, indice, feat_type in zip(
                row[self.feature_input_list].values,
                self.feature_input_indice_list,
                self.feature_type_list,
            ):
                raw_data = np.load(feat_loc, allow_pickle=True)
                if self.end == -1:
                    end = raw_data.shape[0]
                else:
                    end = self.end
                raw_data = raw_data.reshape(raw_data.shape[0], -1)[self.start : end, indice]
                if feat_type == "global":
                    # repeat five times
                    raw_data = np.repeat(raw_data, 5, axis=1).reshape(raw_data.shape[0], -1, 5).transpose(0, 2, 1)
                else:
                    raw_data = raw_data.reshape(raw_data.shape[0], 5, -1)

                feature_trajectory.append(raw_data)
            feature_trajectory = np.concatenate(feature_trajectory, axis=2).reshape(raw_data.shape[0], -1)

            if self.symmetrize:
                self.feature_trajectories.extend(get_symmetrized_data([feature_trajectory], self.multimer))
            else:
                self.feature_trajectories.append(feature_trajectory)
        self.data_collected = True

    def start_analysis(self):
        os.makedirs(self.filename, exist_ok=True)
        raise NotImplementedError("Should be overridden by subclass")
        if not self.data_collected:
            self.gather_feature_matrix()

    def clustering_with_deeptime(self, n_clusters, meaningful_tic=None, updating=False, max_iter=1000):
        # if attr tica_output is None, then tica is not performed
        if not hasattr(self, "tica_output"):
            raise ValueError("TICA output not available")
        self.n_clusters = n_clusters

        if meaningful_tic is None:
            meaningful_tic = np.arange(self.tica_output[0].shape[1])
        self.meaningful_tic = meaningful_tic
        print("Meaningful TICs are", meaningful_tic)

        self.tica_output_filter = [np.asarray(output)[:, meaningful_tic] for output in self.tica_output]

        if not (os.path.isfile(self.cluster_filename + "_deeptime.pickle")) or updating:
            print("Start new cluster analysis")
            self.rerun_msm = True
            self.kmean = KMeans(
                n_clusters=self.n_clusters,
                init_strategy="kmeans++",
                max_iter=max_iter,
                n_jobs=24,
                progress=tqdm,
            )
            self.cluster = self.kmean.fit(self.tica_output_filter).fetch_model()
            self.cluster_dtrajs = [
                self.cluster.transform(tic_output_traj) for tic_output_traj in self.tica_output_filter
            ]
            self.cluster_centers = self.cluster.cluster_centers
            self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)

            os.makedirs(self.filename, exist_ok=True)
            pickle.dump(self.cluster, open(self.cluster_filename + "_deeptime.pickle", "wb"))
        else:
            print("Loading old cluster analysis")

            self.cluster = pickle.load(open(self.cluster_filename + "_deeptime.pickle", "rb"))
            self.cluster_dtrajs = [
                self.cluster.transform(tic_output_traj) for tic_output_traj in self.tica_output_filter
            ]
            self.cluster_centers = self.cluster.cluster_centers
            self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)

    def assigning_cluster(self, cluster_dtrajs, n_clusters=None):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        self.cluster_dtrajs = cluster_dtrajs
        self.meaningful_tic = "all"
        self.dtrajs_concatenated = np.concatenate(self.cluster_dtrajs)

    @staticmethod
    def bayesian_msm_from_traj(cluster_dtrajs, lagtime, n_samples, only_timescales):
        counts = TransitionCountEstimator(
            lagtime=lagtime,
            count_mode="effective",
        ).fit_fetch(cluster_dtrajs, n_jobs=1)
        if only_timescales:
            return BayesianMSM(n_samples=n_samples).fit_fetch(counts).timescales()
        return BayesianMSM(n_samples=n_samples).fit_fetch(counts)

    def get_its(
        self,
        cluster="deeptime",
        lag_max=200,
        n_samples=1000,
        n_jobs=10,
        only_timescales=False,
        updating=False,
        joblib_kwargs={},  # kwargs for joblib
    ) -> ImpliedTimescales:
        """Get ITS for MSMs

        Parameters
        ----------
        cluster : str, optional
            Which cluster to use, by default "deeptime". Currently
            only deeptime is supported.
        lag_max : int, optional
            Maximum lagtime, by default 200
        n_samples : int, optional
            Number of samples for Bayesian MSM, by default 1000
        n_jobs : int, optional
            Number of jobs for parallelization, by default 10
        only_timescales : bool, optional
            Only return timescales, by default False. Currently
            not implemented.
        updating : bool, optional
            Whether to update existing ITS, by default False
        joblib_kwargs : dict, optional
            kwargs for joblib, by default {}
        """
        if only_timescales:
            raise NotImplementedError("Only timescales not implemented yet")
        if cluster == "deeptime":
            if (
                not (os.path.isfile(self.cluster_filename + f"_deeptime_its_{n_samples}.pickle"))
                or updating
                or self.rerun_msm
            ):
                print("Start new ITS analysis")
                lagtimes = np.linspace(1, lag_max / self.dt, 10).astype(int)
                print("Lagtimes are", lagtimes * self.dt, "ns")
                if n_jobs != 1:
                    with tqdm_joblib(tqdm(desc="ITS", total=10)) as progress_bar:
                        models = Parallel(n_jobs=n_jobs, **joblib_kwargs)(
                            delayed(self.bayesian_msm_from_traj)(
                                self.cluster_dtrajs, lagtime, n_samples, only_timescales
                            )
                            for lagtime in lagtimes
                        )
                else:
                    models = []
                    for lagtime in tqdm(lagtimes, desc="lagtime", total=len(lagtimes)):
                        counts = TransitionCountEstimator(lagtime=lagtime, count_mode="effective").fit_fetch(
                            self.cluster_dtrajs
                        )
                        models.append(BayesianMSM(n_samples=n_samples).fit_fetch(counts))

                print("Keep ITS analysis")
                self.its_models = models
                self.its = implied_timescales(models)

                pickle.dump(
                    self.its,
                    open(
                        self.cluster_filename + f"_deeptime_its_{n_samples}.pickle",
                        "wb",
                    ),
                )
                pickle.dump(
                    self.its_models,
                    open(
                        self.cluster_filename + f"_deeptime_its_models_{n_samples}.pickle",
                        "wb",
                    ),
                )
            else:
                print("Loading old ITS analysis")

                self.its = pickle.load(
                    open(
                        self.cluster_filename + f"_deeptime_its_{n_samples}.pickle",
                        "rb",
                    )
                )
                self.its_models = pickle.load(
                    open(
                        self.cluster_filename + f"_deeptime_its_models_{n_samples}.pickle",
                        "rb",
                    )
                )

        return self.its

    def plot_its(self, n_its=10, step="ns"):
        fig, ax = plt.subplots(figsize=(18, 10))
        plot_implied_timescales(self.its, n_its=n_its, ax=ax)
        ax.set_yscale("log")
        ax.set_title("Implied timescales")

        if step == "ns":
            ax.set_xlabel(f"lag time (x {self.dt} ns)")
            ax.set_ylabel(f"timescale (x {self.dt} ns)")
        else:
            ax.set_xlabel(f"lag time (step)")
            ax.set_ylabel(f"timescale (step)")
        plt.show()

    def get_ck_test(
        self,
        n_states,
        lag,
        mlags=6,
        n_jobs=6,
        n_samples=20,
        only_timescales=False,
        updating=False,
        joblib_kwargs={},  # kwargs for joblib
    ):
        if only_timescales:
            raise NotImplementedError("Only timescales not implemented yet")
        if (
            not updating
            and not self.rerun_msm
            and (os.path.isfile(self.cluster_filename + f"_deeptime_cktest.pickle"))
        ):
            print("Loading old CK test")
            self.ck_test = pickle.load(open(self.cluster_filename + f"_deeptime_cktest.pickle", "rb"))

        if (n_states, lag, mlags) not in self.ck_test or updating:
            print("CK models building")
            model = BayesianMSM(n_samples=n_samples).fit_fetch(
                TransitionCountEstimator(lagtime=lag, count_mode="effective").fit_fetch(self.cluster_dtrajs)
            )
            lagtimes = np.arange(1, mlags + 1) * lag
            print("Estimating lagtimes", lagtimes)

            if n_jobs != 1:
                with tqdm_joblib(tqdm(desc="ITS", total=len(lagtimes))) as progress_bar:
                    test_models = Parallel(n_jobs=n_jobs, **joblib_kwargs)(
                        delayed(self.bayesian_msm_from_traj)(
                            self.cluster_dtrajs, lagtime, n_samples, only_timescales
                        )
                        for lagtime in lagtimes
                    )
            else:
                test_models = []
                for lagtime in tqdm(lagtimes, desc="lagtime", total=len(lagtimes)):
                    counts = TransitionCountEstimator(lagtime=lagtime, count_mode="effective").fit_fetch(
                        self.cluster_dtrajs
                    )
                    test_models.append(BayesianMSM(n_samples=n_samples).fit_fetch(counts))
            print("Start CK test")
            self.ck_test[n_states, lag, mlags] = {
                "model": model,
                "ck_test": model.ck_test(test_models, n_states, progress=tqdm),
                "models": test_models,
            }
            pickle.dump(
                self.ck_test,
                open(self.cluster_filename + f"_deeptime_cktest.pickle", "wb"),
            )

        return plot_ck_test(self.ck_test[n_states, lag, mlags]["ck_test"])

    def get_maximum_likelihood_msm(self, lag, cluster="deeptime", updating=False):
        self.msm_lag = lag
        if cluster == "deeptime":
            if (
                not (os.path.isfile(self.cluster_filename + f"_deeptime_max_msm_{lag}.pickle"))
                or updating
                or self.rerun_msm
            ):
                print("Start new MSM analysis")
                self.msm = MaximumLikelihoodMSM(reversible=True, stationary_distribution_constraint=None)
                self.msm.fit(self.cluster_dtrajs, lagtime=lag)
                self.msm_model = self.msm.fetch_model()
                pickle.dump(
                    self.msm,
                    open(self.cluster_filename + f"_deeptime_max_msm_{lag}.pickle", "wb"),
                )
                pickle.dump(
                    self.msm_model,
                    open(
                        self.cluster_filename + f"_deeptime_max_msm_model_{lag}.pickle",
                        "wb",
                    ),
                )
            else:
                print("Loading old MSM analysis")
                self.msm = pickle.load(open(self.cluster_filename + f"_deeptime_max_msm_{lag}.pickle", "rb"))
                self.msm_model = pickle.load(
                    open(
                        self.cluster_filename + f"_deeptime_max_msm_model_{lag}.pickle",
                        "rb",
                    )
                )
            self.trajectory_weights = self.msm_model.compute_trajectory_weights(self.cluster_dtrajs)
        return self.msm_model

    def get_bayesian_msm(self, lag, n_samples=100, cluster="deeptime", updating=False):
        self.msm_lag = lag
        if cluster == "deeptime":
            if (
                not (os.path.isfile(self.cluster_filename + f"_deeptime_bayesian_msm_{lag}.pickle"))
                or updating
                or self.rerun_msm
            ):
                print("Start new MSM analysis")
                self.counts = TransitionCountEstimator(lagtime=lag, count_mode="effective").fit_fetch(
                    self.cluster_dtrajs
                )
                self.msm = BayesianMSM(n_samples=n_samples).fit(self.counts)

                self.msm_model = self.msm.fetch_model()

                from deeptime.markov.tools.analysis import stationary_distribution

                pi_samples = []
                traj_weights_samples = []
                for sample in self.msm_model.samples:
                    pi_samples.append(stationary_distribution(sample.transition_matrix))
                    traj_weights_samples.append(sample.compute_trajectory_weights(self.cluster_dtrajs))

                self.pi_samples = np.array(pi_samples, dtype=object)
                self.traj_weights_samples = np.array(traj_weights_samples, dtype=object)

                self.stationary_distribution = np.mean(self.pi_samples, axis=0)
                self.pi = self.stationary_distribution
                self.trajectory_weights = np.mean(self.traj_weights_samples, axis=0)

                pickle.dump(
                    self.counts,
                    open(
                        self.cluster_filename + f"_deeptime_bayesian_counts_{lag}.pickle",
                        "wb",
                    ),
                )
                pickle.dump(
                    self.msm,
                    open(
                        self.cluster_filename + f"_deeptime_bayesian_msm_{lag}.pickle",
                        "wb",
                    ),
                )
                pickle.dump(
                    self.msm_model,
                    open(
                        self.cluster_filename + f"_deeptime_bayesian_msm_model_{lag}.pickle",
                        "wb",
                    ),
                )

            else:
                print("Loading old MSM analysis")
                self.counts = pickle.load(
                    open(
                        self.cluster_filename + f"_deeptime_bayesian_counts_{lag}.pickle",
                        "rb",
                    )
                )
                self.msm = pickle.load(
                    open(
                        self.cluster_filename + f"_deeptime_bayesian_msm_{lag}.pickle",
                        "rb",
                    )
                )
                self.msm_model = pickle.load(
                    open(
                        self.cluster_filename + f"_deeptime_bayesian_msm_model_{lag}.pickle",
                        "rb",
                    )
                )

                from deeptime.markov.tools.analysis import stationary_distribution

                pi_samples = []
                traj_weights_samples = []
                for sample in self.msm_model.samples:
                    pi_samples.append(stationary_distribution(sample.transition_matrix))
                    traj_weights_samples.append(sample.compute_trajectory_weights(self.cluster_dtrajs))

                self.pi_samples = np.array(pi_samples, dtype=object)
                self.traj_weights_samples = np.array(traj_weights_samples, dtype=object)

                self.stationary_distribution = np.mean(self.pi_samples, axis=0)
                self.pi = self.stationary_distribution
                self.trajectory_weights = np.mean(self.traj_weights_samples, axis=0)
        return self.msm_model

    def get_connected_msm(self):
        if self.msm_model is None:
            raise ValueError("No MSM model found")
        msm_model = self.msm_model
        self.active_set = msm_model.prior.count_model.states_to_symbols(msm_model.prior.count_model.states)
        self.inactive_set = list(
            set(range(msm_model.prior.count_model.n_states_full)).difference(set(self.active_set))
        )
        assignment = self.assignments_concat
        cluster_rank = self.cluster_rank_concat
        self.stat_rank_mapping = {}
        for i in range(cluster_rank.max() + 1):
            self.stat_rank_mapping[i] = assignment[np.where(cluster_rank == i)[0][0]]

    def get_connected_pcca_msm(self):
        if self.msm_model is None:
            raise ValueError("No MSM model found")
        if self.pcca is None:
            raise ValueError("No PCCA model found")

        cluster_rank = self.cluster_rank_concat
        assignment = self.assignments_concat

        if self.inactive_set != []:
            # get connected indices
            disconnection_indices = []
            for inactive_stat in self.inactive_set:
                disconnection_indices.append(np.where(cluster_rank == inactive_stat)[0])
            self.disconnection_indices = np.asarray(list(set(np.concatenate(disconnection_indices))), dtype=int)
            self.connected_indices = np.setdiff1d(np.arange(len(cluster_rank)), self.disconnection_indices)

            self.cluster_rank_connected_concat = rankdata(cluster_rank[self.connected_indices], method="dense") - 1
            self.assignment_connected = assignment[self.connected_indices]
        else:
            self.connected_indices = np.arange(len(cluster_rank))
            self.cluster_rank_connected_concat = cluster_rank
            self.assignment_connected = assignment

        self.stat_rank_mapping_connected = {}
        for i in range(cluster_rank.max() + 1):
            self.stat_rank_mapping_connected[i] = self.assignment_connected[np.where(cluster_rank == i)[0][0]]

        # metastable_traj = [self.pcca.assignments[c_traj] for c_traj in cluster_dtrajs]
        self.metastable_concat = self.pcca.assignments[self.cluster_rank_connected_concat]

    @property
    def transformer(self):
        return self._transformer

    @transformer.setter
    def transformer(self, transformer):
        self._transformer = transformer

    def transform_feature_trajectories(
        self,
        md_dataframe,
        start=0,
        end=-1,
        system_exclusion=[],
        symmetrized=True,
        subunit=False,
    ) -> List[np.ndarray]:
        """
        Map new feature trajectories to the MSM transformer space.
        Note the feature trajectories will expanded based on the multimer size if symmetrized
        is set to `True`.
        i.e. if multimer is 5, the feature trajectories will be expanded to 5 times.
        """
        if subunit:
            if not callable(getattr(self.transformer, "transform_subunit", None)):
                raise ValueError("Transformer does not have a transform_subunit method")

        if end == -1:
            end = md_dataframe.dataframe.shape[0]

        mapped_feature_trajectories = []
        feature_df = md_dataframe.get_feature(self.feature_input_list, in_memory=False)
        for system, row in tqdm(feature_df.iterrows(), total=feature_df.shape[0]):
            if system in system_exclusion:
                continue
            feature_trajectory = []
            for feat_loc, indice, feat_type in zip(
                row[self.feature_input_list].values,
                self.feature_input_indice_list,
                self.feature_type_list,
            ):
                raw_data = np.load(feat_loc, allow_pickle=True)
                raw_data = raw_data.reshape(raw_data.shape[0], -1)[start:end, indice]
                if feat_type == "global":
                    # repeat five times
                    raw_data = np.repeat(raw_data, 5, axis=1).reshape(raw_data.shape[0], -1, 5).transpose(0, 2, 1)
                else:
                    raw_data = raw_data.reshape(raw_data.shape[0], 5, -1)

                feature_trajectory.append(raw_data)

            feature_trajectory = np.concatenate(feature_trajectory, axis=2).reshape(raw_data.shape[0], -1)
            if symmetrized:
                feature_trajectories = get_symmetrized_data([feature_trajectory], self.multimer)
            else:
                feature_trajectories = [feature_trajectory]
            for single_traj in feature_trajectories:
                if subunit:
                    mapped_feature_trajectories.append(self.transformer.transform_subunit(single_traj))
                else:
                    mapped_feature_trajectories.append(self.transformer.transform(single_traj))

        return mapped_feature_trajectories

    @property
    def filename(self):
        if self.feature_input_list == []:
            feature_list_str = ""
        else:
            feature_list_str = "__".join(
                [
                    f"{feature}_{len(feat_n)}"
                    for feature, feat_n in zip(self.feature_input_list, self.feature_input_info_list)
                ]
            )

        if self.end == -1:
            return (
                f"{self.md_dataframe.filename}/msmfile/{self.prefix}/{self.lag}/{self.start}/{feature_list_str}/"
            )
        else:
            return f"{self.md_dataframe.filename}/msmfile/{self.prefix}/{self.lag}/{self.start}_{self.end}/{feature_list_str}/"

    @property
    def cluster_filename(self):
        return (
            self.filename
            + "cluster"
            + str(self.n_clusters)
            + "_tic"
            + "_".join([str(m_tic) for m_tic in self.meaningful_tic])
            + "_"
        )


# TODO: check msm model from basemodel information
"""
from pydantic import BaseModel
class MSMMetaData(BaseModel):
    create_time: Optional[datetime] = None
    id: int = 0
    name = "MSM"
    lag: int
    start: int
    end: int
    multimer: int
    symmetrize: bool
    system_exclusion: Optional[List[int]] = []
    interval: int
    prefix: Optional[str] = None
    feature_input_info_list: Optional[List[str]] = []
"""
