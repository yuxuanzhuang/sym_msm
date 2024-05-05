"""
@author: yuxuanzhuang
"""

from scipy.spatial.distance import cdist
import numpy as np
from sklearn.decomposition import KernelPCA

def permute_stride_matrix(traj_subsystem_feats,
                          n_subsystems,
                          stride=1):
    """
    Augment the feature matrix by permuting the features of each subsystem
    according to the symmetry of the system.

    Parameters
    ----------
    traj_feats : np.ndarray of shape (n_frames, n_features)
        The feature array of the trajectory, where the features of each
        subsystem are grouped together.

    n_subsystems : int
        The number of subsystems in the system.

    stride : int
        The stride to use when permuting the features.

    Returns
    -------
    augmented_feats : np.ndarray of shape (n_frames * n_subsystems, n_features)
        The augmented feature matrix.

    Examples
    --------
    import numpy as np
    from sym_msm.decomposition.sym_mds import permute_stride_matrix

    subsystem_value = np.random.normal(1, 0.1, 10)
    print(subsystem_value.shape)
    > (10,)
    traj_feats = np.tile(subsystem_value, (5,1)).T
    print(traj_feats.shape)
    > (10, 5)

    symmetric_traj_aug = permute_stride_matrix(traj_feats, 5)
    print(symmetric_traj_aug.shape)
    > (50, 5)
    """
    n_feats_per_subsys = traj_subsystem_feats.shape[1] // n_subsystems
    augmented_feats = []
    for i in range(n_subsystems):
        augmented_feats.append(np.roll(traj_subsystem_feats,
                                       n_feats_per_subsys * i, 1)[::stride])
    augmented_feats = np.concatenate(augmented_feats, axis=0)
    return augmented_feats


def _get_dissimilarity_matrix(traj_subsystem_feats):
    n_feats_per_subsys = traj_subsystem_feats.shape[1] // 5
    dist = np.inf * np.ones((traj_subsystem_feats.shape[0],
                             traj_subsystem_feats.shape[0]))
    for i in range(0,5):
        dist_new = cdist(
                traj_subsystem_feats,
                np.roll(traj_subsystem_feats,
                        n_feats_per_subsys * i, 1),
                metric='euclidean')
        dist = np.min([dist, dist_new], axis=0)
        del dist_new
    return dist


def symmetric_mds(input_arr, n_components=2):
    """
    Perform symmetry-aware MDS on the input array.

    Parameters
    ----------
    input_arr : np.ndarray
        The input array to perform MDS on.
        Note that the input array should already be augmented
        by permuting 
    n_components : int
        The number of components to reduce the input array to.
    
    Returns
    -------
    transformed_space : np.ndarray
        The transformed space of the input array.

    Examples
    --------
    import numpy as np
    from sym_msm.decomposition.sym_mds import symmetric_mds
    n_samples = 10
    subsystem_value = np.random.normal(1, 0.1, n_samples)
    traj = np.tile(subsystem_value, (5,1)).T

    # change one subsystem to be different
    asymmetric_value = np.random.normal(1, 0.1, n_samples)
    traj[:, 0] = asymmetric_value
    traj_aug = permute_stride_matrix(traj, 5)

    transformed_space = symmetric_mds(traj_aug, 2)
    print(transformed_space.shape)
    > (50, 2)
    """

    dis_mat = _get_dissimilarity_matrix(input_arr)

    dis_mat =(dis_mat + dis_mat.T) / 2
    dis_mat = dis_mat ** 2
    dis_mat *= -1/2
    # metric MDS on dissimilarity matrix
    kpca = KernelPCA(n_components=n_components,
                     kernel='precomputed')
    transformed_space = kpca.fit_transform(dis_mat)
    return transformed_space