import argparse

import pytorch_lightning as pl
from kaizen.args.dataset import augmentations_args, dataset_args
from kaizen.args.utils import additional_setup_linear, additional_setup_pretrain
from kaizen.args.continual import continual_args
from kaizen.methods import METHODS
from kaizen.utils.checkpointer import Checkpointer
from kaizen.distillers import DISTILLERS
from kaizen.distiller_factories import DISTILLER_FACTORIES
from .utils import strtobool

try:
    from kaizen.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

def parse_args_pretrain(input_args=None) -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)
    continual_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args(input_args)

    # add model specific args
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    # add distiller args
    if temp_args.distiller_library == "default":
        if temp_args.distiller is not None:
            parser = DISTILLERS[temp_args.distiller]().add_model_specific_args(parser)
    elif temp_args.distiller_library == "factory":
        if temp_args.distiller is not None:
            parser = DISTILLER_FACTORIES[temp_args.distiller](class_tag="feature_extractor").add_model_specific_args(parser)
        if temp_args.distiller_classifier is not None:
            parser = DISTILLER_FACTORIES[temp_args.distiller_classifier](class_tag="classifier").add_model_specific_args(parser)

    # add checkpoint and auto umap args
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--auto_umap", action="store_true")
    temp_args, _ = parser.parse_known_args(input_args)

    # optionally add checkpointer and AutoUMAP args
    if temp_args.save_checkpoint:
        parser = Checkpointer.add_checkpointer_args(parser)

    if _umap_available and temp_args.auto_umap:
        parser = AutoUMAP.add_auto_umap_args(parser)

    # parse args
    args, unknown_args = parser.parse_known_args(input_args)
    print("Unknown args:", unknown_args)

    # prepare arguments with additional setup
    additional_setup_pretrain(args)

    return args


def parse_args_linear() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds and arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["linear"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--num_tasks", type=int, default=2)
    SPLIT_STRATEGIES = ["class", "data", "domain"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, required=True)
    parser.add_argument("--domain", type=str, default=None)

    # add checkpointer args (only if logging is enabled)
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args, unknown_args = parser.parse_known_args()
    print("Unknown Args:", unknown_args)
    additional_setup_linear(args)

    return args

def parse_args_eval() -> argparse.Namespace:
    """Parses trained model, dataset, pytorch lightning, evaluation specific and additional args.

    First adds and arg for the pretrained model, then adds dataset, pytorch lightning
    and eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--evaluation_mode", choices=["linear_eval", "classifier_eval", "online_classifier_eval"], default="linear_eval", type=str)
    parser.add_argument("--linear_classifier_training_data_source", choices=["all_tasks", "seen_tasks", "current_task"], default="all_tasks", type=str)

    # Memory Bank/Replay args
    parser.add_argument("--replay", type=strtobool, default=False)
    parser.add_argument("--replay_proportion", type=float, default=1.0)
    parser.add_argument("--replay_memory_bank_size", type=int, default=None)
    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["full_model"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--task_idx", type=int) # For linear classifier training only
    SPLIT_STRATEGIES = ["class", "data", "domain"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, required=True)
    parser.add_argument("--domain", type=str, default=None)

    # add checkpointer args (only if logging is enabled)
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args, unknown_args = parser.parse_known_args()
    print("Unknown Args:", unknown_args)
    additional_setup_linear(args)
    if len(args.classifier_layers) < len(args.online_eval_classifier_layers):
        args.classifier_layers = args.online_eval_classifier_layers

    return args
