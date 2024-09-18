import argparse
import logging
from src import constants
from src.models import phi_mat_distiller


logging.basicConfig(
    level=logging.INFO,
    format="[%asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distill", help="type of distillation", default="mat", type=str
    )
    parser.add_argument(
        "--model",
        help="the model to distill",
        default="microsoft/phi-1_5",
        type=str,
    )

    args = parser.parse_args()

    if args.distill == "mat":
        phi_mat_distiller.training_loop(args.model, "datasets/python-github-code")
