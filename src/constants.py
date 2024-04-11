import torch
import matplotlib.pyplot as plt
import os
import logging
from tqdm_logger import TqdmToLogger

from dotenv import load_dotenv
import os

# Automatically finds and loads the .env file
load_dotenv()

plt.style.use("ggplot")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

DATA_DIR = "data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the STDOUT environment variable
STDOUT = os.environ.get("STDOUT", "False").lower() in ("true", "1", "t")

# Use half precision weights to save memory
IS_HALF_PRECISION = os.environ.get("HALF_PRECISION", "True").lower() in (
    "true",
    "1",
    "t",
)

# half precision in pytorch is only implemented for cuda kernels

MODEL_PRECISION = (
    torch.float16 if IS_HALF_PRECISION and torch.cuda.is_available() else torch.float32
)


if STDOUT:
    # Configure logging to output to stdout
    print("output all logs to stdout")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
else:
    # Configure logging to write to a file
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=DATA_DIR + "main.log",
        filemode="w",
    )

TQDM_OUTPUT = TqdmToLogger(logging.getLogger(), level=logging.INFO)

MODEL_PATHS = {
    "phi1.5": "llm_cache/models--microsoft--phi-1_5",
    "gpt2": "llm_cache/gpt2",
}
