#!/bin/bash
#SBATCH --job-name=mtl-nyu-uniformnonlin
#SBATCH -p zen5_mpi
#SBATCH --cpus-per-task=1
#SBATCH --time=84:00:00

#SBATCH --array=0-1%1
#SBATCH --output=mtl-nyu-unif-lin%A_%a.out
#SBATCH --mail-type=ALL

module load SciPy-bundle/2023.07-gfbf-2023a
module load skorch/0.15.0-foss-2023a-PyTorch-2.1.2
module load scikit-learn/1.3.1-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load Seaborn/0.13.2-gfbf-2023a

set -euo pipefail

PROJECT_DIR=
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKTYPE="${TASKTYPE:-nyuv2_depth_normal}"
BATCH_SIZE="${BATCH_SIZE:-636}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
TRAINING_STEPS="${TRAINING_STEPS:-300000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MODEL_TYPE="${MODEL_TYPE:-relu}"
BACKEND="${BACKEND:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/cluster_runs/depth_normal_sweep-2layer-e-4-h16nonlin}"
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-nyudepthnormal}"
TRACKGRADIENTS="${TRACKGRADIENTS:-true}"
MAKE_PLOTS="${MAKE_PLOTS:-true}"
SAVE_WEIGHT_HISTORY="${SAVE_WEIGHT_HISTORY:-false}"
Y1_NORMALIZE_TARGETS="${Y1_NORMALIZE_TARGETS:-1}"
CSV_LOG_INTERVAL="${CSV_LOG_INTERVAL:-1}"


NYUV2_ROOT=
MODE="${MODE:-train}"
AUGMENTATION="${AUGMENTATION:-false}"
INPUT_DOWNSAMPLE_FACTOR="${INPUT_DOWNSAMPLE_FACTOR:-32}"
Y1_DOWNSAMPLE_FACTOR="${Y1_DOWNSAMPLE_FACTOR:-16}"
Y2_DOWNSAMPLE_FACTOR="${Y2_DOWNSAMPLE_FACTOR:-16}"
WHITEN_TASK_INPUTS="${WHITEN_TASK_INPUTS:-false}"

MAIN_TASK="${MAIN_TASK:-1}"
WEIGHTING_MODE="none"
GRADIENT_SURGERY="${GRADIENT_SURGERY:-none}"

SEEDS_STR="${SEEDS:-12,30,40,50}"
GRADNORM_ALPHAS_STR="${GRADNORM_ALPHAS:-1.5}"
GRADNORM_LRS_STR="${GRADNORM_LRS:-0.01}"

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a ALPHA_LIST <<< "$GRADNORM_ALPHAS_STR"
read -r -a LR_LIST <<< "$GRADNORM_LRS_STR"

NUM_SEEDS=${#SEED_LIST[@]}
NUM_ALPHAS=${#ALPHA_LIST[@]}
NUM_LRS=${#LR_LIST[@]}
TOTAL_RUNS=$(( NUM_SEEDS * NUM_ALPHAS * NUM_LRS ))
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( ARRAY_ID >= TOTAL_RUNS )); then
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_ID} exceeds TOTAL_RUNS=${TOTAL_RUNS}."
  exit 1
fi


SEED_INDEX=$(( ARRAY_ID / (NUM_ALPHAS * NUM_LRS) ))
REM=$(( ARRAY_ID % (NUM_ALPHAS * NUM_LRS) ))
ALPHA_INDEX=$(( REM / NUM_LRS ))
LR_INDEX=$(( REM % NUM_LRS ))

SEED="${SEED_LIST[$SEED_INDEX]}"
GRADNORM_ALPHA="${ALPHA_LIST[$ALPHA_INDEX]}"
GRADNORM_LR="${LR_LIST[$LR_INDEX]}"

RUN_NAME="${OUTPUT_PREFIX_BASE}_${TASKTYPE}_h${HIDDEN_DIM}_lr${LEARNING_RATE}_steps${TRAINING_STEPS}_seed${SEED}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

CMD=(
  "$PYTHON_BIN" "run_cluster_experiment.py"
  --tasktype "$TASKTYPE"
  --batch-size "$BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --training-steps "$TRAINING_STEPS"
  --learning-rate "$LEARNING_RATE"
  --model-type "$MODEL_TYPE"
  --backend "$BACKEND"
  --seed "$SEED"
  --random "$SEED"
  --weighting-mode "$WEIGHTING_MODE"
  --gradient-surgery "$GRADIENT_SURGERY"
  --main-task "$MAIN_TASK"
  --gradnorm-alpha "$GRADNORM_ALPHA"
  --gradnorm-lr "$GRADNORM_LR"
  --csv-log-interval "$CSV_LOG_INTERVAL"
  --output-dir "$OUTPUT_DIR"
  --output-prefix "$RUN_NAME"
  --nyuv2-root "$NYUV2_ROOT"
  --mode "$MODE"
  --input-downsample-factor "$INPUT_DOWNSAMPLE_FACTOR"
  --y1-downsample-factor "$Y1_DOWNSAMPLE_FACTOR"
  --y2-downsample-factor "$Y2_DOWNSAMPLE_FACTOR"
)

if [[ "$Y1_NORMALIZE_TARGETS" == "1" ]]; then
  CMD+=(--y1-normalize-targets)
fi

if [[ "$TRACKGRADIENTS" == "true" ]]; then
  CMD+=(--trackgradients)
else
  CMD+=(--no-trackgradients)
fi

if [[ "$MAKE_PLOTS" == "true" ]]; then
  CMD+=(--make-plots)
else
  CMD+=(--no-make-plots)
fi

if [[ "$SAVE_WEIGHT_HISTORY" == "true" ]]; then
  CMD+=(--save-weight-history)
else
  CMD+=(--no-save-weight-history)
fi

if [[ "$AUGMENTATION" == "true" ]]; then
  CMD+=(--augmentation)
else
  CMD+=(--no-augmentation)
fi

if [[ "$WHITEN_TASK_INPUTS" == "true" ]]; then
  CMD+=(--whiten-task-inputs)
else
  CMD+=(--no-whiten-task-inputs)
fi

if [[ -n "${SPLIT_JSON:-}" ]]; then
  CMD+=(--split-json "$SPLIT_JSON")
fi

echo "Array job ${ARRAY_ID}/${TOTAL_RUNS}: task=${TASKTYPE} seed=${SEED} gradnorm_alpha=${GRADNORM_ALPHA} gradnorm_lr=${GRADNORM_LR}"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
