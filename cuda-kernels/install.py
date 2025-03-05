import argparse
import site
import shutil
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

torch.cuda.is_available() or sys.exit("CUDA not available")
base_dir = Path(__file__).resolve().parent


def install(project_name: str):
    project_dir = base_dir / project_name
    build_dir = base_dir / "../build" / project_name
    build_dir.mkdir(parents=True, exist_ok=True)

    # find all .cu or .cpp files in project_dir
    sources = [
        str(f) for f in project_dir.iterdir() if f.suffix in [".cpp", ".cu"]
    ]

    # build the sources
    cpp_extension.load(
        name=project_name,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=["-lcudart"],
        build_directory=build_dir,
        with_cuda=True,
        verbose=True,
    )

    # find the compiled .so file
    so_file = build_dir / f"{project_name}.so"
    if not so_file.exists():
        print(f"Error: Compiled .so file not found at '{so_file}'.")
        return 1

    # copy the .so file to site-packages
    site_packages_dir = Path(site.getsitepackages()[0])
    dest_so_file = site_packages_dir / f"{project_name}.so"

    try:
        shutil.copy2(so_file, dest_so_file)
        print(f"Installed {project_name}.so to {dest_so_file}")
    except Exception as e:
        print(f"Error installing to site-packages: {e}")
        return 1

    # check if we can import the installed module
    try:
        __import__(project_name)
        print(f"Successfully verified import of '{project_name}'")
    except ImportError as e:
        print(f"Warning: Could not verify import of '{project_name}': {e}")
        print("Installation may still have succeeded; try importing manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    args = parser.parse_args()
    sys.exit(install(args.project))
