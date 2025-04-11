import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup


# TODO: why is this here?
def init_git_submodules():
    """Initialize git submodules if .git directory exists and --init-git flag is set"""
    if "--init-git" in sys.argv:
        if Path(".git").exists():
            print("Initializing git submodules...")
            try:
                subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
                print("Git submodules initialized successfully")
            except subprocess.CalledProcessError as e:
                print(f"Git submodule initialization failed: {e}")
                print("Continuing without git submodules...")
        else:
            print("No .git directory found, skipping git submodule initialization")
    else:
        print("Skipping git submodule initialization (use --init-git to enable)")


# Handle dependency conflicts by defining dependencies with proper constraints
install_requires = [
    "click>=8.0.0",
    "pybids>=0.15.1",
    "nipype>=1.8.5",
    "nibabel>=5.0.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "prov>=2.0.0",
    "rdflib==6.3.2",
    "rapidfuzz>=2.0.0",
    "pytest>=7.0.0",
    "xlrd>=2.0.0",
    "neurdflib>=0.1.0",
]

# These dependencies will be installed with --no-deps to avoid conflicts
extras_require = {
    "conflicting": [
        "PyLD==2.0.4",  # This conflicts with pynidm's requirement of pyld<2.0
        "pynidm==4.1.0",  # This requires pyld<2.0
    ],
}

# TODO: remove
# Check if we're being called with a container build command
# if len(sys.argv) > 1 and sys.argv[1] in ["docker", "singularity", "containers"]:
#     command = sys.argv[1]
#     # Remove the custom argument so setup() doesn't see it
#     sys.argv.pop(1)
#
#     if command == "docker":
#         build_docker()
#     elif command == "singularity":
#         # Check for custom output path in the next argument
#         output_path = None
#         if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
#             output_path = sys.argv.pop(1)
#         build_singularity(output_path)
#     elif command == "containers":
#         build_docker()
#         build_singularity()
#
#     # Exit if we were just building containers
#     if len(sys.argv) == 1:
#         sys.exit(0)

# TODO: why is it here, i think it is always called?
# Initialize git submodules only if explicitly requested
init_git_submodules()

setup(
    name="bids-freesurfer",
    version="0.1.0",
    description="BIDS App for FreeSurfer with NIDM Output",
    author="ReproNim Team",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    url="https://github.com/yourusername/bids-freesurfer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "bids-freesurfer=src.run:cli",
        ],
    },
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
)