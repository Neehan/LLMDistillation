# Skip Layer Distillation for LLMs

This project implements a skip layer distillation technique for Large Language Models (LLMs), specifically targeting Phi-3 models. The technique trains the model to perform well even when certain layers are skipped during inference, potentially leading to faster inference with minimal impact on model quality.

## Project Overview

Skip layer distillation works by:

1. Training each layer to be "skippable" - the model learns to perform well both with and without the layer
2. Focusing on every other layer, starting from the second-to-last layer
3. Using knowledge distillation to transfer knowledge from the full model to the skippable version

This approach allows for runtime decisions about which layers to skip, creating a flexible trade-off between speed and quality.

## Requirements

- Python 3.12
- PyTorch
- Transformers
- Accelerate
- Other dependencies listed in `requirements.txt`

## Project Structure

```
.
├── data/                      # Data directory
│   ├── datasets/              # Raw datasets cache
│   ├── encodings_train/       # Tokenized training data
│   ├── encodings_test/        # Tokenized test data
│   ├── logs/                  # Training logs
│   └── llm_cache/             # Model cache
├── src/                       # Source code
│   ├── models/                # Model-specific implementations
│   │   └── phi_3_skip_layer_distiller.py
│   ├── base_distiller.py      # Base distillation class
│   ├── skip_layer_distiller.py # Skip layer distillation implementation
│   ├── constants.py           # Global constants
│   ├── setup.py               # Data preparation script
│   ├── training_loop.py       # Main training loop
│   └── utils.py               # Utility functions
├── setup.sh                   # Environment setup script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Hugging Face Authentication (Required)

This project uses the Phi-3 model from Microsoft, which requires authentication with Hugging Face. Before running the setup script:

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Request access to the Phi-3 model at [huggingface.co/microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
4. Add your token to the `.env` file after setup: `HF_TOKEN=your_token_here`

Without a valid token, you won't be able to access the Phi-3 models for training.

### Automated Setup

Run the setup script to install all dependencies and prepare the data:

```bash
bash setup.sh
```

This will:
1. Create necessary directories
2. Set up a default .env file
3. Install Miniconda if not already installed
4. Install CUDA if not already installed (Linux only)
5. Create a conda environment named "llm"
6. Install all required packages from requirements.txt
7. Prepare training and test data with visible progress bars

After running the setup script, make sure to add your Hugging Face token to the `.env` file if you haven't already.

### Manual Installation

If you prefer to install manually:

```bash
# Create and activate conda environment
conda create --name llm python=3.12
conda activate llm

# Install PyTorch (adjust based on your platform)
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install -r requirements.txt

# Prepare the data
python -m src.setup
```

## Configuration

The project uses environment variables for configuration, which can be set in the `.env` file:

- `STDOUT`: Set to 1 to output logs to stdout instead of a file
- `HAS_CUDA`: Set to 1 if CUDA is available (auto-detected in code)
- `USE_HALF_PRECISION`: Set to 1 to use half precision (bfloat16/float16) to save memory
- `TEST_ENV`: Set to 1 to enable test environment with additional logging and smaller dataset
- `HF_TOKEN`: Your Hugging Face authentication token (required for accessing Phi-3 models)

## Data Preparation

The data preparation is automatically handled by the setup script. It:

1. Downloads the dataset from HuggingFace (codeparrot/github-code-clean)
2. Tokenizes the data using the Phi-3 tokenizer (or fallback to GPT-2 if no HF token is provided)
3. Splits the data into training and test sets
4. Saves the tokenized data in chunks to disk

By default, it prepares:
- 50,000 training samples in chunks of 5,000 examples each
- 10,000 test samples

If `TEST_ENV=1` is set in your `.env` file, it will use a smaller dataset:
- 10,000 training samples in chunks of 2,000 examples each
- 3,000 test samples

To manually reprocess the data:

```bash
conda activate llm
python -m src.setup --force
```

You can customize the data preparation with additional arguments:
- `--train_samples`: Number of training samples (default: 50,000)
- `--test_samples`: Number of test samples (default: 10,000)
- `--train_chunk_size`: Size of each training data chunk (default: 5,000)
- `--max_length`: Maximum sequence length (default: 4,096)

## Training

To train the model:

```bash
conda activate llm
python -m src.models.phi_3_skip_layer_distiller
```

### Command Line Arguments

- `--model`: The model to distill (default: "microsoft/Phi-3-mini-128k-instruct")
- `--lr`: Learning rate (default: 1e-5)
- `--num_epochs`: Number of training epochs per layer (default: 3)
- `--max_seq_len`: Maximum sequence length (default: 512)
- `--batch_size`: Batch size for training (default: 4)

## How It Works

1. The code loads a teacher model (e.g., Phi-3-mini)
2. It creates a student model as a copy of the teacher
3. For each layer (every other layer, starting from the second-to-last):
   - It wraps the layer with a "skippable" version that can be toggled on/off
   - It trains the model to perform well both with and without the layer
   - It uses knowledge distillation to minimize the difference between teacher and student outputs
4. The final model is saved and can be used with layers selectively skipped

## Data Handling

The code uses a streaming approach to handle large datasets efficiently:
- Data is loaded in chunks to avoid memory issues
- Each chunk is processed and then discarded
- This allows training on datasets larger than available memory

## Output

The distilled model is saved to `data/llm_cache/student_model/`. This model can be loaded like any other Hugging Face model, but with the added ability to skip layers during inference.

Training logs are saved to `data/logs/` with timestamped filenames.

## Memory Optimization

The code includes several memory optimization techniques:
- Fully Sharded Data Parallel (FSDP) for distributed training
- Half-precision training (bfloat16 or float16)
- Garbage collection between layer training
- Streaming data loading
- Optional flash attention for CUDA devices (automatically enabled when CUDA is available)

## License

[MIT License](LICENSE)
