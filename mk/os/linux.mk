# Linux build, GPU guard and daemon lifecycle hooks.
# The Laptop calls these through scripts/remote.py.

EXE :=
TARGET_TRIPLE ?= x86_64-unknown-linux-gnu

# $(call todo,<what>): a recipe line that fails with a pointer to this file.
todo = @echo "error: $1 is not implemented on Linux yet (see mk/os/linux.mk)" >&2; exit 1

KERNEL_BUILD = bash kernel/build.sh
KERNEL_TEST = bash kernel/build.sh build --test
KERNEL_BUILD_INPUTS := $(wildcard kernel/build.sh)

GPU_STATUS = bash mk/linux/gpu.sh status
GPU_GUARD = bash mk/linux/gpu.sh guard $(GPU_THRESHOLD_MIB)
GPU_PROFILE = python3 scripts/gpu_profile.py --artifact "$(ARTIFACT)" $(GPU_PROFILE_ARGS)

SERVER_CTL_ARGS = --pid '$(SERVER_PID)' --log '$(SERVER_LOG)' --url '$(SERVER_URL)' --timeout '$(READY_TIMEOUT)'
SERVER_START = python3 mk/linux/server.py start $(SERVER_CTL_ARGS) -- '$(SERVER_BIN)' $(SERVER_FLAGS)
SERVER_STOP = python3 mk/linux/server.py stop $(SERVER_CTL_ARGS) $(if $(filter 1,$(FORCE)),--force)
SERVER_STATUS = python3 mk/linux/server.py status $(SERVER_CTL_ARGS)
# sh: start the server and vite in one process group, `trap 'kill 0' INT TERM EXIT`.
DEV_UI = $(call todo,run-ui / dev-ui)

# Version and changelog helpers.
# mk/changelog.sh is git and gh, which are the same on either host.
VERSION_TOOL = bash mk/version.sh
CHANGELOG_TOOL = bash mk/changelog.sh

OS_REQUIRED_TOOLS := python3
OS_OPTIONAL_TOOLS := cmake ninja nvidia-smi
OS_TOOL_HINT := CUDA=1 only: CUDA Toolkit under CUDA_HOME, CMake, Ninja -- README Prerequisites
OS_DOCTOR = echo "cuda:"; \
  cuda="$${CUDA_HOME:-$${CUDA_PATH:-/usr/local/cuda}}"; \
  if [ -x "$$cuda/bin/nvcc" ]; then echo "  ok       nvcc ($$cuda)"; \
  else echo "  optional nvcc not under $$cuda (set CUDA_HOME; CUDA=1 only)"; fi;
