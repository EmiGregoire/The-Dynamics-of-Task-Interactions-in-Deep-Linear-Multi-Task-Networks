#!/bin/bash
#SBATCH --job-name=mtl-net-MNISTgpualiseeds
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:35:00
#SBATCH --output=mtl-net-%j.out

#SBATCH --mail-type=ALL


module load SciPy-bundle/2023.07-gfbf-2023a
module load skorch/0.15.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

set -euo pipefail

PROJECT_DIR=
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKTYPE="${TASKTYPE:-multimnist}"
BATCH_SIZE="${BATCH_SIZE:-50000}"
HIDDEN_DIM="${HIDDEN_DIM:-20}"
TRAINING_STEPS="${TRAINING_STEPS:-400000}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"
MODEL_TYPE="${MODEL_TYPE:-linear}"
BACKEND="${BACKEND:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_runs}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-slurm}"


NYUV2_ROOT=
WEIGHTING_MODE="${WEIGHTING_MODE:-none}"
SHIFT_RANGE="${SHIFT_RANGE:-4}"
ALIGNMENT="${ALIGNMENT:-true}"
GRADIENT_SURGERY="${GRADIENT_SURGERY:-none}"
GRADNORM_ALPHA="${GRADNORM_ALPHA:-1.5}"
GRADNORM_LR="${GRADNORM_LR:-0.025}"
OUTPUT_PREFIX=multimnist_2seed20_shift4_alireal_uniform_h20gpuj
SEED="${SEED:-20}"
MULTIMNIST_SOURCE=torchvision
WHITEN_TASK_INPUTS="${WHITEN_TASK_INPUTS:-true}"


cd "$PROJECT_DIR"

CMD=(
  "$PYTHON_BIN" "run_cluster_experiment.py"
  --tasktype "$TASKTYPE"
  --batch-size "$BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --training-steps "$TRAINING_STEPS"
  --learning-rate "$LEARNING_RATE"
  --model-type "$MODEL_TYPE"
  --backend "$BACKEND"
  --output-dir "$OUTPUT_DIR"
  --output-prefix "$OUTPUT_PREFIX"
  --seed "$SEED"
)

if [[ "$TASKTYPE" == multimnist ]]; then
  MULTIMNIST_SOURCE="${MULTIMNIST_SOURCE:-torchvision}"
  CMD+=(--multimnist-source "$MULTIMNIST_SOURCE")
  CMD+=(--shift-range "$SHIFT_RANGE")

  if [[ "$ALIGNMENT" == true ]]; then
    CMD+=(--alignment)
  else
    CMD+=(--no-alignment)
  fi


  if [[ "$MULTIMNIST_SOURCE" == torchvision ]]; then
    MNIST_ROOT="${MNIST_ROOT:-$PROJECT_DIR/data}"
    CMD+=(--mnist-root "$MNIST_ROOT")
  else
    MULTIMNIST_DATA_NPY="${MULTIMNIST_DATA_NPY:?Set MULTIMNIST_DATA_NPY for npy input}"
    MULTIMNIST_LABELS_NPY="${MULTIMNIST_LABELS_NPY:?Set MULTIMNIST_LABELS_NPY for npy input}"
    CMD+=(--multimnist-data-npy "$MULTIMNIST_DATA_NPY" --multimnist-labels-npy "$MULTIMNIST_LABELS_NPY")
  fi
  
else
  NYUV2_ROOT=
  INPUT_DOWNSAMPLE_FACTOR="${INPUT_DOWNSAMPLE_FACTOR:-1}"
  Y1_DOWNSAMPLE_FACTOR="${Y1_DOWNSAMPLE_FACTOR:-4}"
  Y2_DOWNSAMPLE_FACTOR="${Y2_DOWNSAMPLE_FACTOR:-4}"

  CMD+=(
    --nyuv2-root "$NYUV2_ROOT"
    --input-downsample-factor "$INPUT_DOWNSAMPLE_FACTOR"
    --y1-downsample-factor "$Y1_DOWNSAMPLE_FACTOR"
    --y2-downsample-factor "$Y2_DOWNSAMPLE_FACTOR"
  )

  if [[ -n "${SPLIT_JSON:-}" ]]; then
    CMD+=(--split-json "$SPLIT_JSON")
  fi
fi
if [[ "$WHITEN_TASK_INPUTS" == "true" ]]; then
  CMD+=(--whiten-task-inputs)
else
  CMD+=(--no-whiten-task-inputs)
fi


echo "Running: ${CMD[*]}"
"${CMD[@]}"
