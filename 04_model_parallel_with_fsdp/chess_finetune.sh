#!/bin/bash -l
#SBATCH --job-name=chess_finetune  # create a short name for your job
#SBATCH --output=%x-%j_%n.out      # file to write stdout
#SBATCH --nodes=2                  # node count
#SBATCH --ntasks-per-node=2        # total number of tasks across all nodes
#SBATCH --cpus-per-task=4          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G                 # cpu memory per node
#SBATCH --gpus-per-node=2          # number of gpus per node
#SBATCH --time=01:00:00            # total run time limit (HH:MM:SS)

spack load py-transformers py-torchdata jq

# You can override the default values of these parameters by adding `TOTAL_BATCH_SIZE=... sbatch chess_finetune.sh`
# The default values are given after the minus (i.e., 64 and 1) and are used if the variable is empty
total_batch_size=${TOTAL_BATCH_SIZE:-64} # total batch size per optimization
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-1} # batch size per device

# Read CUDA_VISIBLE_DEVICES to detect the number of GPUs
num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")

# Compute the gradient accumulation steps given a total batch size and a batch size per device
gradient_accumulation_steps=$(($total_batch_size / $batch_size_per_device / $num_gpus))

echo "Starting torchrun"
function random_port ()
{
    # "What's the easiest way to find an unused local port?", https://unix.stackexchange.com/a/423052
    comm -23 <(seq "1024" "65535" | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1
}
export RDZV_HOST=$(hostname)
export RDZV_PORT=$(random_port)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

srun --distribution=block \
    torchrun \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}" \
    --nnodes=${SLURM_NNODES} \
    --nproc-per-node=$num_gpus \
    chess_finetune.py \
        --batch_size_per_device $batch_size_per_device \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        $@
