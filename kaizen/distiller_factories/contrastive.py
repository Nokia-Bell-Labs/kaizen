import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from kaizen.losses.simclr import simclr_distill_loss_func
from kaizen.args.utils import strtobool

def contrastive_distill_factory(
    Method=object, class_tag="",
    distill_current_key="z", distill_frozen_key="frozen_z", output_dim=256
):
    distill_lamb_name = f"{class_tag}_distill_lamb"
    distill_proj_hidden_dim_name = f"{class_tag}_distill_proj_hidden_dim"
    distill_temperature_name = f"{class_tag}_distill_temperature"
    distill_no_predictior_name = f"{class_tag}_distill_no_predictior"

    distill_predictor_name = f"{class_tag}_distill_predictor"
    class ContrastiveDistillWrapper(Method):
        def __init__(self, **kwargs):
            distill_lamb: float = kwargs.pop(distill_lamb_name)
            distill_proj_hidden_dim: int = kwargs.pop(distill_proj_hidden_dim_name)
            distill_temperature: float = kwargs.pop(distill_temperature_name)
            distill_no_predictior = kwargs.pop(distill_no_predictior_name)
            super().__init__(**kwargs)

            setattr(self, distill_lamb_name, distill_lamb)
            setattr(self, distill_proj_hidden_dim_name, distill_proj_hidden_dim)
            setattr(self, distill_temperature_name, distill_temperature)
            setattr(self, distill_no_predictior_name, distill_no_predictior)
            if distill_no_predictior:
                setattr(self, distill_predictor_name, nn.Identity())
            else:
                setattr(self, distill_predictor_name, nn.Sequential(
                    nn.Linear(output_dim, distill_proj_hidden_dim),
                    nn.BatchNorm1d(distill_proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(distill_proj_hidden_dim, output_dim),
                ))
            
        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group(f"contrastive_{class_tag}_distiller")

            parser.add_argument(f"--{distill_lamb_name}", type=float, default=1)
            parser.add_argument(f"--{distill_proj_hidden_dim_name}", type=int, default=2048)
            parser.add_argument(f"--{distill_temperature_name}", type=float, default=0.2)
            parser.add_argument(f"--{distill_no_predictior_name}", type=strtobool, default=False)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": getattr(self, distill_predictor_name).parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out[distill_current_key]
            frozen_z1, frozen_z2 = out[distill_frozen_key]

            p1 = getattr(self, distill_predictor_name)(z1)
            p2 = getattr(self, distill_predictor_name)(z2)

            distill_loss = (
                simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2, getattr(self, distill_temperature_name))
                + simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2, getattr(self, distill_temperature_name))
            ) / 2

            self.log(f"train_{class_tag}_contrastive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            out["loss"] += getattr(self, distill_lamb_name) * distill_loss
            return out

    return ContrastiveDistillWrapper