export NVIDIA_PYTORCH_VERSION=24.04
export CUDNN_PATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12
export CUDA_PATH=/global/homes/k/klhhhhh/cuda12.4
export CPATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12/include:$CPATH
export PATH=/global/homes/k/klhhhhh/cuda12.4/bin:$PATH
export CUDA_HOME=/global/homes/k/klhhhhh/cuda12.4
export LD_LIBRARY_PATH=/global/homes/k/klhhhhh/cuda12.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/global/common/software/nersc/pe/gpu/gnu/openmpi/5.0.0/lib:$LD_LIBRARY_PATH

export CUDACXX=/global/homes/k/klhhhhh/cuda12.4/bin/nvcc

export PATH=$HOME/gcc-12.3/bin:$PATH
export LD_LIBRARY_PATH=$HOME/gcc-12.3/lib64:$LD_LIBRARY_PATH

export HF_HOME=/pscratch/sd/k/klhhhhh/hf_cache_lm_eval
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/models
export HF_DATASETS_TRUST_REMOTE_CODE=1

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0