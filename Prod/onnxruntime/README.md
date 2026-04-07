# ONNX Runtime Notebooks

This repository contains short notebooks on how to run ONNX Runtime with the TensorRT provider.

The examples focus on practical setup, quick validation, and common TensorRT-related errors.

## Files

- `onnxrt_colab.ipynb` shows how to export a model to ONNX and run it in Colab.
- `onnxrt_bench.ipynb` provides a small benchmark for CPU, CUDA, and TensorRT.
- `onnxrt_paths.ipynb` explains library path issues and how to fix `LD_LIBRARY_PATH`.
- `onnxrt_cuda.ipynb` explains CUDA version compatibility for ONNX Runtime and TensorRT.

## Setup

Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate

```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download example model weights

```bash
wget https://ml.gan4x4.ru/wb/cv/models/resnet18_onnx.zip
unzip resnet18_onnx.zip -d model
rm resnet18_onnx.zip
```

Or generate it using the [1_onnxrt_colab.ipynb](1_onnxrt_colab.ipynb) notebook file.

You also need an NVIDIA GPU environment if you want to run the CUDA or TensorRT examples.

## Common issue

If TensorRT is installed inside Python `site-packages`, ONNX Runtime may fail to find the required native libraries.
In that case, add the TensorRT and cuDNN library directories to `LD_LIBRARY_PATH` and restart the notebook kernel.

For stable local development, installing TensorRT system-wide is usually simpler than using a Python-only install.


-------------------------------------------------------------

[Video](https://youtu.be/3FPS1jadWDU): 
