import torch
import matplotlib.pyplot as plt
import os
import logging
from tqdm_logger import TqdmToLogger

plt.style.use("ggplot")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

DATA_DIR = "data/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=DATA_DIR + "main.log",
    filemode="w",
)

TQDM_OUTPUT = TqdmToLogger(logging.getLogger(), level=logging.INFO)
