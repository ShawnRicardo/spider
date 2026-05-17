# Contributing to SPIDER

Thank you for your interest in contributing to SPIDER (Scalable Physics-Informed DExterous Retargeting)! We welcome contributions from the community to help improve our physics-based retargeting framework.

## Getting Started

Before you begin, please review our project structure and architectural guidelines.

### Environment Setup

We recommend using `uv` for dependency management, though `conda` is also supported. The project requires **Python 3.12+**.

**Option 1: Using uv (Recommended)**
```bash
uv python install 3.12
uv sync --python 3.12
pip install --ignore-requires-python --no-deps -e .
````

**Option 2: Using conda**

```bash
conda create -n spider python=3.12
conda activate spider
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Development Workflow

### Code Style and Formatting

We use **Ruff** for all linting and formatting. Please ensure your code passes all checks before submitting a Pull Request.

  * **Style:** Google-style docstrings
  * **Line Length:** 88 characters

**Commands:**

```bash
# Run linter
ruff check .

# Auto-fix simple issues
ruff check --fix .

# Format code
ruff format .
```

### Running Tests

We have specific tests for individual simulator backends. Please run the relevant tests for the modules you modify.

```bash
# Test MJWP (Mujoco Warp) simulator
uv run spider/simulators/mjwp_test.py

# Test DexMachina simulator
uv run spider/simulators/dexmachina_test.py

# Test HDMI simulator
uv run spider/simulators/hdmi_test.py
```

## How to Contribute

1.  **Fork the repository** and create your branch from `main`.
2.  **Install dependencies** using the setup instructions above.
4.  **Run tests and linter.** Ensure `ruff check .` passes and all relevant simulator tests run successfully.
5.  **Submit a Pull Request.** Provide a clear description of your changes and link to any relevant issues.

### Adding New Features

  * **New Robots:** Requires adding MJCF assets, updating `spider/config.py` embodiment mappings, and adjusting reward weights.
  * **New Datasets:** Requires a new processor in `spider/process_datasets/` that outputs standard NPZ files.

## License

By contributing, you agree that your contributions will be licensed under the project's existing license.
