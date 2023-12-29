from copy import deepcopy
from typing import Any, Sequence
import torch


def base_frozen_model_factory(MethodClass=object):
    class BaseFrozenModel(MethodClass):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

            self.output_dim = kwargs["output_dim"]
            self.store_model_frozen_copy()

        def store_model_frozen_copy(self):
            self.frozen_encoder = deepcopy(self.encoder)
            self.frozen_projector = deepcopy(self.projector)
            
            for pg in self.frozen_encoder.parameters():
                pg.requires_grad = False
            for pg in self.frozen_projector.parameters():
                pg.requires_grad = False
            if self.classifier_training:
                self.frozen_classifier = deepcopy(self.classifier)
                for pg in self.frozen_classifier.parameters():
                    pg.requires_grad = False
            else:
                self.frozen_classifier = None

        def on_train_start(self):
            super().on_train_start()
            self.store_model_frozen_copy()

        @torch.no_grad()
        def frozen_forward(self, X):
            feats_encoder = self.frozen_encoder(X)
            feats_projector = self.frozen_projector(feats_encoder)
            if self.frozen_classifier is not None:
                logits_classifier = self.frozen_classifier(feats_encoder)
            else:
                logits_classifier = None
            return feats_encoder, feats_projector, logits_classifier

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            _, (X1, X2), _ = batch[f"task{self.current_task_idx}"]
            if "replay" in batch:
                *_, (X1R, X2R), _ = batch["replay"]
                X1 = torch.cat([X1, X1R])
                X2 = torch.cat([X2, X2R])

            out = super().training_step(batch, batch_idx)

            frozen_feats1, frozen_z1, frozen_logits1 = self.frozen_forward(X1)
            frozen_feats2, frozen_z2, frozen_logits2 = self.frozen_forward(X2)

            out.update({
                "frozen_feats": [frozen_feats1, frozen_feats2], 
                "frozen_z": [frozen_z1, frozen_z2],
                "frozen_logits": [frozen_logits1, frozen_logits2]
            })
            return out

    return BaseFrozenModel
