# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from packaging import version
from onnxruntime.capi.build_and_package_info import pytorch_version as build_pytorch_version


################################################################################
# All global constant goes here, before ORTModule is imported ##################
################################################################################
ONNX_OPSET_VERSION = 12
MINIMUM_RUNTIME_PYTORCH_VERSION_STR = '1.8.1'

# Verify minimum PyTorch version is installed before proceding to ONNX Runtime initialization
try:
    import torch
    runtime_pytorch_version = version.parse(torch.__version__.split('+')[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:
        raise RuntimeError(
            f'ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR}, '
            f'but version {torch.__version__} was found instead.')
except:
    raise(f'PyTorch {MINIMUM_RUNTIME_PYTORCH_VERSION_STR} must be installed in order to run ONNX Runtime ORTModule frontend!')

# Verify PyTorch installed on the system is compatible with the version used during ONNX Runtime build
if not build_pytorch_version:
    raise RuntimeError('ONNX Runtime was compiled without PyTorch support!')

if runtime_pytorch_version != version.parse(build_pytorch_version.split('+')[0]):
    raise RuntimeError(f'ONNX Runtime was compiled using PyTorch {build_pytorch_version}'
                       f', but PyTorch {runtime_pytorch_version} is installed on your system!')


# PyTorch custom Autograd function support
from ._custom_autograd_function import enable_custom_autograd_support
enable_custom_autograd_support()

# ORTModule must be loaded only after all validation passes
from .ortmodule import ORTModule
