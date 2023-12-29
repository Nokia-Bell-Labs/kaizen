from argparse import ArgumentParser
from .utils import strtobool
DISTILLER_LIBRARIES = ["default", "factory"]

def continual_args(parser: ArgumentParser):
    """Adds continual learning arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """
    # base continual learning args
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--task_idx", type=int, required=True)

    SPLIT_STRATEGIES = ["class", "data", "domain"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, required=True)

    # distillation args
    parser.add_argument("--distiller", type=str, default=None)
    parser.add_argument("--distiller_classifier", type=str, default=None)
    parser.add_argument("--distiller_library", type=str, choices=DISTILLER_LIBRARIES, default=DISTILLER_LIBRARIES[0])

    # Memory Bank/Replay args
    parser.add_argument("--replay", type=strtobool, default=False)
    parser.add_argument("--replay_proportion", type=float, default=1.0)
    parser.add_argument("--replay_memory_bank_size", type=int, default=None)
    parser.add_argument("--replay_batch_size", type=int, default=64)
