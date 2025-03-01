"""
Constants and configuration settings for the LLM distillation project.
Loads environment variables and sets up logging configuration.
"""

import torch
import os
import logging
import time
import platform
from src.tqdm_logger import TqdmToLogger
from dotenv import load_dotenv

# Automatically finds and loads the .env file
load_dotenv()

# Set deterministic algorithms for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# Platform detection
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

# Directory paths
DATA_DIR = "data/"
LOGS_DIR = os.path.join(DATA_DIR, "logs/")

# Hardware detection and configuration
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    os.environ["HAS_CUDA"] = "1"
else:
    os.environ["HAS_CUDA"] = "0"

# Device configuration
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")

# Environment configuration from .env file
# Whether to output logs to stdout instead of a file
STDOUT = os.environ.get("STDOUT", "False").lower() in ("true", "1", "t")
# Whether we're in test environment (enables additional logging)
TEST_ENV = os.environ.get("TEST_ENV", "False").lower() in ("true", "1", "t")

# Use half precision weights to save memory
USE_HALF_PRECISION = os.environ.get("USE_HALF_PRECISION", "True").lower() in (
    "true",
    "1",
    "t",
)

# Set model precision based on environment and hardware availability
if USE_HALF_PRECISION and HAS_CUDA:
    MODEL_PRECISION = torch.bfloat16
elif USE_HALF_PRECISION:
    MODEL_PRECISION = torch.float16
else:
    MODEL_PRECISION = torch.float32

# Generate a timestamp for log files
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_filename = f"training_{timestamp}.log"

# Configure logging
if STDOUT:
    # Configure logging to output to stdout
    print("Output all logs to stdout")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
else:
    # Configure logging to write to a file in the logs directory
    log_file_path = os.path.join(LOGS_DIR, log_filename)
    print(f"Logging to file: {log_file_path}")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file_path,
        filemode="w",
    )

# Configure TQDM progress bar to output to logger
TQDM_OUTPUT = TqdmToLogger(logging.getLogger(), level=logging.INFO)

# Minimum interval between logging in seconds
MIN_INTERVAL_SEC = 2 * 60
