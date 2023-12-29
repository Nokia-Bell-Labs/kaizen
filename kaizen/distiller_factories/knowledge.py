import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from kaizen.args.utils import strtobool

def cross_entropy(preds, targets):
    return -torch.mean(
        torch.sum(F.softmax(targets, dim=-1) * torch.log_softmax(preds, dim=-1), dim=-1)
    )

def knowledge_distill_factory(
    Method=object, class_tag="",
    distill_current_key="z", distill_frozen_key="frozen_z", output_dim=256
):
    distill_lamb_name = f"{class_tag}_distill_lamb"
    distill_proj_hidden_dim_name = f"{class_tag}_distill_proj_hidden_dim"
    distill_temperature_name = f"{class_tag}_distill_temperature"
    distill_no_predictior_name = f"{class_tag}_distill_no_predictior"

    frozen_prototypes_name = f"{class_tag}_frozen_prototypes"
    distill_predictor_name = f"{class_tag}_distill_predictor"
    distill_prototypes_name = f"{class_tag}_distill_prototypes"
    class KnowledgeDistillWrapper(Method):
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
            # TODO: Allow different num_prototypes for different distillers
            # TODO: Verify that these prototypes can be used for classifier distillation
            num_prototypes = kwargs["num_prototypes"]

            setattr(self, frozen_prototypes_name, nn.utils.weight_norm(
                nn.Linear(output_dim, num_prototypes, bias=False)
            ))
            for frozen_pg, pg in zip(
                getattr(self, frozen_prototypes_name).parameters(), self.prototypes.parameters() # TODO: Check this
            ):
                frozen_pg.data.copy_(pg.data)
                frozen_pg.requires_grad = False

            if distill_no_predictior:
                setattr(self, distill_predictor_name, nn.Identity())
            else:
                setattr(self, distill_predictor_name, nn.Sequential(
                    nn.Linear(output_dim, distill_proj_hidden_dim),
                    nn.BatchNorm1d(distill_proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(distill_proj_hidden_dim, output_dim),
                ))
                

            setattr(self, distill_prototypes_name, nn.utils.weight_norm(
                nn.Linear(output_dim, num_prototypes, bias=False)
            ))

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group(f"knowledge_{class_tag}_distiller")

            parser.add_argument(f"--{distill_lamb_name}", type=float, default=1)
            parser.add_argument(f"--{distill_proj_hidden_dim_name}", type=int, default=2048)
            parser.add_argument(f"--{distill_temperature_name}", type=float, default=0.1)
            parser.add_argument(f"--{distill_no_predictior_name}", type=strtobool, default=True)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": getattr(self, distill_predictor_name).parameters()},
                {"params": getattr(self, distill_prototypes_name).parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def on_train_start(self):
            super().on_train_start()

            if self.current_task_idx > 0:
                for frozen_pg, pg in zip(
                    getattr(self, frozen_prototypes_name).parameters(), self.prototypes.parameters()
                ):
                    # TODO: logits and prototypes have different shape
                    frozen_pg.data.copy_(pg.data)
                    frozen_pg.requires_grad = False

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out[distill_current_key]
            frozen_z1, frozen_z2 = out[distill_frozen_key]

            with torch.no_grad():
                frozen_z1 = F.normalize(frozen_z1)
                frozen_z2 = F.normalize(frozen_z2)
                frozen_p1 = getattr(self, frozen_prototypes_name)(frozen_z1) / getattr(self, distill_temperature_name)
                frozen_p2 = getattr(self, frozen_prototypes_name)(frozen_z2) / getattr(self, distill_temperature_name)

            distill_z1 = F.normalize(self.distill_predictor(z1))
            distill_z2 = F.normalize(self.distill_predictor(z2))
            distill_p1 = getattr(self, distill_prototypes_name)(distill_z1) / getattr(self, distill_temperature_name)
            distill_p2 = getattr(self, distill_prototypes_name)(distill_z2) / getattr(self, distill_temperature_name)

            distill_loss = (
                cross_entropy(distill_p1, frozen_p1) + cross_entropy(distill_p2, frozen_p2)
            ) / 2

            self.log(f"train_{class_tag}_knowledge_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            out["loss"] += getattr(self, distill_lamb_name) * distill_loss
            return out

    return KnowledgeDistillWrapper
