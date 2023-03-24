from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset
import numpy as np
from typing import Optional, List


class MultimerTrajectoriesDataset(TrajectoriesDataset):
    """
    A dataset for multimer trajectories.
    Warning: The features in this dataset should be n-fold symmetric.
    """

    def __init__(self, multimer: int, data: List[TrajectoryDataset]):
        self.multimer = multimer
        super().__init__(data)

    @staticmethod
    def from_numpy(lagtime, multimer, data: List[np.ndarray]):
        assert isinstance(data, list)
        assert len(data) > 0 and all(data[0].shape[1:] == x.shape[1:] for x in data), "Shape mismatch!"

        data_new = []
        total_shape = data[0].shape[1]
        per_shape = int(total_shape / multimer)

        for i in range(multimer):
            data_new.extend(
                [
                    np.roll(traj.reshape(traj.shape[0], multimer, per_shape), i, axis=1).reshape(
                        traj.shape[0], total_shape
                    )
                    for traj in data
                ]
            )
        return MultimerTrajectoriesDataset(multimer, [TrajectoryDataset(lagtime, traj) for traj in data_new])


def get_symmetrized_data(data: List[np.ndarray], multimer: int) -> np.ndarray:
    """
    Symmetrize the data.
    """
    assert data[0].shape[1] % multimer == 0
    total_shape = data[0].shape[1]

    per_shape = int(total_shape / multimer)
    data_new = []
    for i in range(multimer):
        data_new.extend(
            [
                np.roll(traj.reshape(traj.shape[0], multimer, per_shape), i, axis=1).reshape(
                    traj.shape[0], total_shape
                )
                for traj in data
            ]
        )
    return data_new
