import os
import types

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet18, resnet50

from kaizen.args.setup import parse_args_eval
from kaizen.methods.full_model import FullModel

try:
    from kaizen.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
from kaizen.methods.linear import LinearModel
from kaizen.methods.multi_layer_classifier import MultiLayerClassifier
from kaizen.utils.classification_dataloader import prepare_data
from kaizen.utils.checkpointer import Checkpointer


def main():
    args = parse_args_eval()

    # split classes into tasks
    tasks = None
    if args.split_strategy == "class":
        assert args.num_classes % args.num_tasks == 0
        torch.manual_seed(args.split_seed)
        tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

    seed_everything(args.global_seed)


    # Build backbone
    if args.encoder == "resnet18":
        backbone = resnet18()
    elif args.encoder == "resnet50":
        backbone = resnet50()
    else:
        raise ValueError("Only [resnet18, resnet50] are currently supported.")

    if args.cifar:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()

    assert (
        args.pretrained_model.endswith(".ckpt")
        or args.pretrained_model.endswith(".pth")
        or args.pretrained_model.endswith(".pt")
    )
    ckpt_path = args.pretrained_model


    state = torch.load(ckpt_path, map_location=None if torch.cuda.is_available() else "cpu")["state_dict"]
    extracted_state = {}
    for k in list(state.keys()):
        if "encoder" in k:
            extracted_state[k.replace("encoder.", "")] = state[k]
    missing_keys_backbone, unexpected_keys_backbone = backbone.load_state_dict(extracted_state, strict=False)
    print("Missing keys - Backbone:", missing_keys_backbone)

    # Build Classifier
    classifier = MultiLayerClassifier(backbone.inplanes, args.num_classes, args.classifier_layers)
    # "--evaluation_mode", choices=["linear_eval", "classifier_eval", "online_classifier_eval"]
    if args.evaluation_mode == "linear_eval":
        is_model_training = True
    elif args.evaluation_mode == "classifier_eval":
        extracted_state = {}
        for k in state:
            if k.startswith("classifier."):
                extracted_state[k.replace("classifier.", "")] = state[k]
        missing_keys_classifier, unexpected_keys_classifier = classifier.load_state_dict(extracted_state, strict=False)
        is_model_training = False
        print("Missing keys - Classifier:", missing_keys_classifier)
        print("Unexpected keys - Classifier:", unexpected_keys_classifier)
    elif args.evaluation_mode == "online_classifier_eval":
        extracted_state = {}
        for k in state:
            if k.startswith("online_eval_classifier."):
                extracted_state[k.replace("online_eval_classifier.", "")] = state[k]
        missing_keys_classifier, unexpected_keys_classifier = classifier.load_state_dict(extracted_state, strict=False)
        is_model_training = False
        print("Missing keys - Classifier:", missing_keys_classifier)
        print("Unexpected keys - Classifier:", unexpected_keys_classifier)

    print(f"Loaded {ckpt_path}")

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        raise NotImplementedError("Dali is not supported")
        MethodClass = types.new_class(
            f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel)
        )
    # else:
    #     MethodClass = LinearModel

    model = FullModel(backbone, classifier=classifier, **args.__dict__, tasks=tasks)

    if is_model_training:
        train_loader, val_loader = prepare_data(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            semi_supervised=args.semi_supervised,
            training_data_source=args.linear_classifier_training_data_source,
            training_num_tasks=args.num_tasks,
            training_tasks=tasks,
            training_task_idx=args.task_idx,
            training_split_strategy=args.split_strategy,
            training_split_seed=args.split_seed,
            replay=args.replay,
            replay_proportion=args.replay_proportion,
            replay_memory_bank_size=args.replay_memory_bank_size
        )
    else:
        _, val_loader = prepare_data(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            semi_supervised=args.semi_supervised,
        )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp" if torch.cuda.is_available() else "cpu",
    )
    if is_model_training:
        if args.dali:
            trainer.fit(model, val_dataloaders=val_loader)
        else:
            trainer.fit(model, train_loader, val_loader)
    else:
        trainer.validate(model, val_loader)

if __name__ == "__main__":
    main()
