#!/bin/bash
#SBATCH -A m4410_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH -c 32
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nankaicslk@gmail.com

# allocate MASTER_PORT
if [ "$SLURM_PROCID" -eq 0 ]; then
  find_free_port() {
    while true; do
      PORT=$(shuf -i 49152-65535 -n 1)
      ss -tuln | grep ":$PORT " > /dev/null
      if [ $? -ne 0 ]; then
        echo $PORT
        return
      fi
    done
  }
  export MASTER_PORT=$(find_free_port)
  echo "Master Port: $MASTER_PORT"
fi

# sync MASTER_PORT
MASTER_PORT=$(srun --ntasks=1 echo $MASTER_PORT)
export MASTER_PORT

# set MASTER_ADDR IP 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1 | xargs getent hosts | awk '{print $1}')

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

export NVIDIA_PYTORCH_VERSION=24.12
export TMPDIR=/pscratch/sd/k/klhhhhh/tmp

srun --ntasks=$SLURM_NTASKS hostname
sleep 5

# nccl
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

srun --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NVIDIA_PYTORCH_VERSION=$NVIDIA_PYTORCH_VERSION,TMPDIR=$TMPDIR \
     bash /global/homes/k/klhhhhh/NeMo-modular-training/modular-training/scripts/gpt/350m_gpt_pretrain.sh
