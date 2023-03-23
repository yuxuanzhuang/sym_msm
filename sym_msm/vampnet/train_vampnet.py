import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
logging.basicConfig(filename="logs.log", level=logging.INFO)
from ENPMDA import MDDataFrame
import itertools
from sym_msm.msm.msm import *
from sym_msm.vampnet import (
    VAMPNETInitializer_Multimer,
    VAMPNet_Multimer_SYM_REV,
)
from sym_msm.vampnet.lobe import *
import sys
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm.notebook import tqdm  # progress bar
from torch.utils.data import DataLoader
import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
torch.set_num_threads(32)

print(f"Using device {device}")

## Train Vampnet

resids_exclusion = (
    list(range(386, 397))
    + list(range(298, 364))
    + list(range(1, 25))
    + list(range(60, 75))
)

def start_training(
    dataframe_name,
    lobe_class,
    resids_exclusion=resids_exclusion,
    prefix="a7_apos_nosym_rev",
    batch_size=8000,
    updating=True,
    overwrite=True,
    lag=50,
    start=100,
    n_epochs=300,
    n_rep=5,
    learning_rate=5e-3,
    early_stopping_patience=20,
    early_stopping_threshold=1e-4,
):
    md_dataframe = MDDataFrame.load_dataframe(dataframe_name)
    msm_obj = VAMPNETInitializer_Multimer(
        md_dataframe=md_dataframe,
        lag=lag,
        multimer=5,
        start=start,
        system_exclusion=[],
        updating=updating,
        symmetrize=False,
        in_memory=True,
        prefix=prefix,
    )
    feat_info = md_dataframe.get_feature_info("ca_distance_10A_2diff")
    feat_ind_exclusion = []
    feat_ind_inclusion = []
    for ind, feat in enumerate(feat_info):
        resid1 = eval(feat.split("_")[1])
        resid2 = eval(feat.split("_")[3])
        if resid1 in resids_exclusion or resid2 in resids_exclusion:
            feat_ind_exclusion.append(ind)
        else:
            feat_ind_inclusion.append(ind)
    msm_obj.add_feature(
        "ca_distance_10A_2diff_reciprocal",
        excluded_indices=feat_ind_exclusion,
        feat_type="subunit",
    )

    total_nfeat = sum([len(feat) for feat in msm_obj.feature_input_info_list])
    print("Total # feats", total_nfeat)
    print("Start collecting data")
    msm_obj.start_analysis()
    dataset = msm_obj.dataset
    n_val = int(len(dataset) * 0.1)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

    pentamer_lobes = []
    multimer_class = getattr(sys.modules[__name__], lobe_class)
    for n_states in range(3, 7):
        pentamer_nstate_lobe = multimer_class(
            data_shape=total_nfeat, multimer=5, n_states=n_states
        )
        pentamer_nstate_lobe = torch.nn.DataParallel(pentamer_nstate_lobe)
        pentamer_lobes.append(pentamer_nstate_lobe)

    class_name = "VAMPNet_Multimer_SYM_REV"
    print("Training VAMPNet Multimer")

    vampnets = [
        VAMPNet_Multimer_SYM_REV(
            multimer=5,
            n_states=pentamer_lobe._modules["module"].n_states,
            rep=rep,
            sym=True,
            lobe=deepcopy(pentamer_lobe).to(device=device),
            score_method="VAMPE",
            learning_rate=learning_rate,
            dtype=np.float64,
            device=device,
        )
        for pentamer_lobe, rep in itertools.product(pentamer_lobes, range(n_rep))
    ]

    for i, vampnet in enumerate(vampnets):
        rep = i % n_rep

        if (
            os.path.isfile(
                f"{msm_obj.filename}{class_name}/epoch_{n_epochs}_state_{vampnet.n_states}_rep_{rep}.lobe"
            )
            and not overwrite
        ):
            vampnets[i] = pickle.load(
                open(
                    f"{msm_obj.filename}{class_name}/epoch_{n_epochs}_state_{vampnet.n_states}_rep_{rep}.lobe",
                    "rb",
                )
            )
            continue

        vampnet.fit(
            loader_train,
            n_epochs=n_epochs,
            validation_loader=loader_val,
            progress=tqdm,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        os.makedirs(msm_obj.filename + class_name, exist_ok=True)

        vampnet.device = torch.device("cpu")
        vampnet._lobe = vampnet._lobe.module.to('cpu')
        vampnet._lobe.eval()
        vampnet._lobe_timelagged = vampnet._lobe_timelagged.module.to('cpu')
        vampnet._lobe_timelagged.eval()
        vampnet.save(folder=msm_obj.filename, n_epoch=n_epochs, rep=rep)

        print(f"# state {vampnet.n_states}, rep {rep}, index {i} finished")

    for vampnet in vampnets:
        print(
            f"state: {vampnet.n_states},"
            f"rep: {vampnet.rep}, \n"
            f"score: {np.max(vampnet.train_scores.T[1]):4f}, "
            f"glob score: {np.max(vampnet.train_scores_full.T[1]):4f}, "
            f"sym score: {np.max(vampnet.train_scores_deg.T[1]):4f}",
        )


def main():
    # set up the argument parser
    parser = argparse.ArgumentParser(description="Run VAMPNet on a dataset")
    parser.add_argument(
        "--dataframe_name",
        type=str,
        default="./a7_apos_feature/a7_apos_feature_md_dataframe",
        help="name of the dataframe to use",
    )
    parser.add_argument(
        "--lobe_class",
        type=str,
        default="MultimerNet_200",
        help="name of the lobe class to use",
    )
    parser.add_argument(
        "--resids_exclusion",
        type=list,
        default=resids_exclusion,
        help="resids to exclude from the dataset",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="a7_apos_nosym_rev",
        help="prefix for the output files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8000, help="batch size for training"
    )
    parser.add_argument(
        "--updating", type=bool, default=True, help="whether to update the model"
    )
    parser.add_argument("--lag", type=int, default=50, help="lag time for the VAMPNet")
    parser.add_argument(
        "--start", type=int, default=100, help="start epoch for the VAMPNet"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=300, help="number of epochs for the VAMPNet"
    )
    parser.add_argument(
        "--n_rep", type=int, default=10, help="number of repetitions for the VAMPNet"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="learning rate for the VAMPNet",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="early stopping patience for the VAMPNet",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=1e-4,
        help="early stopping threshold for the VAMPNet",
    )

    # parse the arguments
    args = parser.parse_args()

    print(args)
    start_training(**vars(args))


# main
if __name__ == "__main__":
    main()