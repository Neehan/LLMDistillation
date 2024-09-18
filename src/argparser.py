import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format="[%asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lr",
    help="learning rate for the optimizer",
    default=0.0004,
    type=float,
)
parser.add_argument(
    "--num_epochs",
    help="number of epochs for training",
    default=1,
    type=int,
)
parser.add_argument(
    "--max_seq_len",
    help="maximum sequence length for training",
    default=2048,
    type=int,
)

parser.add_argument(
    "--distill_dim",
    help="maximum sequence length for training",
    default=4096,
    type=int,
)
parser.add_argument(
    "--batch_size",
    help="size of the batches for training",
    default=1,
    type=int,
)


parser.add_argument(
    "--model",
    help="the model to distill",
    default="microsoft/phi-1_5",
    type=str,
)
