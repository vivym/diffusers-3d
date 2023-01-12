import os
import sys
import sysconfig
import subprocess
from pathlib import Path
from setuptools import find_packages, setup

import torch

from tools.setup_helpers.cmake import CMake
from tools.setup_helpers.env import IS_WINDOWS

PROJECT_NAME = "diffusers-3d"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "Diffusers-3D"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 13], "Requires PyTorch >= 1.13"


def get_version() -> str:
    cwd = Path(__file__).parent

    with open(cwd / "version.txt") as f:
        version = f.readline().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        sha = "Unknown"

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    # write version.py
    with open(cwd / PACKAGE_NAME / "version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")

    return version


def build_extension(version: str) -> None:
    cmake = CMake()

    # CMAKE: full path to python library
    if IS_WINDOWS:
        cmake_python_library = "{}/libs/python{}.lib".format(
            sysconfig.get_config_var("prefix"),
            sysconfig.get_config_var("VERSION"))
        # Fix virtualenv builds
        # TODO: Fix for python < 3.3
        if not os.path.exists(cmake_python_library):
            cmake_python_library = "{}/libs/python{}.lib".format(
                sys.base_prefix,
                sysconfig.get_config_var("VERSION"))
    else:
        cmake_python_library = "{}/{}".format(
            sysconfig.get_config_var("LIBDIR"),
            sysconfig.get_config_var("INSTSONAME"))

    cmake.generate(
        version=version,
        cmake_python_library=cmake_python_library,
        cmake_torch_library=torch.utils.cmake_prefix_path,
        rerun=False,
    )
    cmake.build()


if __name__ == "__main__":
    version = get_version()

    print(f"Building {PROJECT_NAME}-{version}")

    build_extension(version)

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Ming Yang",
        author_email="ymviv@qq.com",
        url=f"https://github.com/vivym/{PROJECT_NAME}",
        download_url=f"https://github.com/vivym/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.9",
        install_requires=[
            "pytorch_lightning",
            "jsonargparse[signatures]",
            "wandb",
            "rich",
            "einops",
            "torchmetrics",
         ],
    )
