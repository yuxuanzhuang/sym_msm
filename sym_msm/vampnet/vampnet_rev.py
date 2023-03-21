import numpy as np

import torch

from typing import Optional, List

from tqdm.notebook import tqdm  # progress bar
from ..deepmsm.deepmsm import *

from typing import Optional, Union, Callable, Tuple
from deeptime.util.torch import disable_TF32, map_data, multi_dot
from deeptime.decomposition.deep import vamp_score, vampnet_loss
from .vampnet import VAMPNet_Multimer, VAMPNet_Multimer_AUG
from .score import vamp_score_sym, vamp_score_rev


class VAMPNet_Multimer_REV(VAMPNet_Multimer_AUG):
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

        loss_value = -vamp_score_rev(
            x_0,
            x_t,
            method=self.score_method,
            epsilon=self.epsilon,
            mode=self.score_mode,
        )

        loss_value.backward()
        self.optimizer.step()

        if self.sym:
            score_value_full = vamp_score_rev(
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
                reversible=True,
                method=self.score_method,
                epsilon=self.epsilon,
                mode=self.score_mode,
            )
        else:
            score_value_full = None
            score_value_deg = None
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

    def validate(self, validation_data: Tuple[torch.Tensor]) -> torch.Tensor:
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
                score_value = vamp_score_rev(
                    val_aug,
                    val_t_aug,
                    method=self.score_method,
                    mode=self.score_mode,
                    epsilon=self.epsilon,
                )
                if self.sym:
                    score_value_full = vamp_score_rev(
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
                        reversible=True,
                        method=self.score_method,
                        mode=self.score_mode,
                        epsilon=self.epsilon,
                    )
                    return score_value, score_value_full, score_value_deg

                return score_value
