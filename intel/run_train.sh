#!/usr/bin/bash
#PBS -j oe

cd $PBS_O_WORKDIR/..

# Detect system from PBS
SYSTEM=""
if [[ ${PBS_O_HOST} == *"aurora"* ]]; then
    SYSTEM="aurora"
elif [[ ${PBS_O_HOST} == *"polaris"* ]]; then
    SYSTEM="polaris"
else
    echo "Unknown system ${SYSTEM}!"
    exit 1
fi

NUMPROCS=${PPN:-12}
NUMNODES=$(cat ${PBS_NODEFILE} | wc -l)
export WORLD_SIZE=$((NUMNODES * NUMPROCS))

LLAMA_CONFIG=${LLAMA_CONFIG:-llama3_8b}
# TODO: check if valid Llama model
CONFIG_FILE="./torchtitan/models/llama3/train_configs/${LLAMA_CONFIG}.toml"

# Loads environment to run the model
DEFAULT_ENV_TO_LOAD=""
PT_CONFIG=${PT_CONFIG:-pt+ipex}
if [ "$PT_CONFIG" == "pt+ipex" ]; then
    DEFAULT_ENV_TO_LOAD="torch_ipex_latest"
elif [[ "$PT_CONFIG" == "pt" ]]; then
    DEFAULT_ENV_TO_LOAD="torch_latest"
else
    echo "Unknown environment to load ${PT_CONFIG}!"
    exit 1
fi

CCL_CONFIG=${CCL_CONFIG:-basekit}
if [ "$CCL_CONFIG" != "basekit" ] && [ "$CCL_CONFIG" != "master" ]; then
    echo "Unknown CCL config to load ${CCL_CONFIG}. Supported are basekit and master."
    exit 1
fi
echo "[Intel] Using oneCCL from ${CCL_CONFIG}" |& tee ${LOG_FILE}

# Do check for valid PyTorch config

ENV_TO_LOAD=${ENV:-${DEFAULT_ENV_TO_LOAD}}
ENV_REL_PATH=./intel/envs/${SYSTEM}/${ENV_TO_LOAD}.env
source ${ENV_REL_PATH} ccl-${CCL_CONFIG}
ENV_FULL_PATH=$(realpath ${ENV_REL_PATH})
ENV_NAME=$(basename $(realpath ${ENV_FULL_PATH}) .env)

if [ "$SYSTEM" == "aurora" ] || [ "$SYSTEM" == "polaris" ]; then
    IFS='.' read -ra ADDR <<< "`cat $PBS_NODEFILE | head -1`"
    export MASTER_ADDR=$ADDR".hsn.cm.${SYSTEM}.alcf.anl.gov"
elif [[ "$SYSTEM" == "borealis" ]]; then
    export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
fi
export MASTER_PORT=29500

TODAY=$(date '+%Y-%m-%d')
TIMENOW=$(date '+%H:%M')
TIMESTAMP=${TODAY}_${TIMENOW}
PBS_JOBNUM="$( cut -d '.' -f 1 <<< "${PBS_JOBID}" )"

LOG_DIR=./intel/outputs/logs/${SYSTEM}/${LLAMA_CONFIG}/${ENV_NAME}/${TODAY}/${PBS_JOBNUM}
LOG_FILE_SUFFIX=${LOG_DIR}/${LLAMA_CONFIG}_${SYSTEM}_${USER}_${NUMNODES}n${NUMPROCS}ppn_${PT_CONFIG}_${PBS_JOBNUM}pbs_${TIMESTAMP}_train
LOG_FILE=${LOG_FILE_SUFFIX}.txt
ENV_FILE=${LOG_FILE_SUFFIX}.env
TOML_FILE=${LOG_FILE_SUFFIX}.toml

mkdir -p ${LOG_DIR}
cp ${ENV_REL_PATH} ${ENV_FILE}
cp ${CONFIG_FILE} ${TOML_FILE}

echo "[Intel] Running ${LLAMA_CONFIG} on ${SYSTEM^} system using ${NUMNODES} nodes with ${NUMPROCS} processes per node" |& tee ${LOG_FILE}
echo "[Intel] Environment loaded from file ${ENV_FULL_PATH}" |& tee -a ${LOG_FILE}

DATASET_PATH=""
if [ "$LLAMA_CONFIG" != "debug_model" ]; then
    if [[ "$SYSTEM" == "aurora" ]]; then
        DATASET_PATH="/flare/Aurora_deployment/sgoswami/datavol/mldata/hf/c4"
    elif [[ "$SYSTEM" == "polaris" ]]; then
        DATASET_PATH="/eagle/datasets/allenai/c4"
    fi
fi

export PYTHONPATH="./":${PYTHONPATH}

CHECKPOINT_TO_LOAD=${CHECKPOINT_TO_LOAD:-""}
if [[ ! -z ${CHECKPOINT_TO_LOAD} ]]; then
    CHECKPOINT_FLAG="--checkpoint.initial_load_path=${CHECKPOINT_TO_LOAD} --checkpoint.no_initial_load_model_weights_only"
    echo "[Intel] Resuming training from checkpoint ${CHECKPOINT_TO_LOAD}" |& tee -a ${LOG_FILE}
else
    CHECKPOINT_FLAG=""
fi

if [[ "$SYSTEM" == "aurora" ]]; then
    source ./intel/envs/${SYSTEM}/common.env
    MPIEXEC_CMD="mpiexec \
                 --envall \
                 --pmi=pmix \
                 --np ${WORLD_SIZE} \
                 --ppn ${NUMPROCS} \
                 --line-buffer \
                 --cpu-bind=${AURORA_CPU_BINDINGS} "
elif [[ "$SYSTEM" == "polaris" ]]; then
    MPIEXEC_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True mpiexec \
                 --envall \
                 --np ${WORLD_SIZE} \
                 --ppn ${NUMPROCS} \
                 --line-buffer \
                 --cpu-bind depth \
                 -d 16 "
fi

ONLY_RANK_ZERO_OUTPUT=${ONLY_RANK_ZERO_OUTPUT:-on}
if [ "$ONLY_RANK_ZERO_OUTPUT" == "ON" ] || [ "$ONLY_RANK_ZERO_OUTPUT" == "on" ]; then
    ONLY_RANK_ZERO_OUTPUT_SCRIPT="./intel/helpers/only_rank_zero_output.sh"
else
    ONLY_RANK_ZERO_OUTPUT_SCRIPT=""
fi
SET_RANK_DEPS_SCRIPT="./intel/helpers/set_ranks_deps.sh"
CONFIG_FILE_FLAG="--job.config_file=${TOML_FILE}"
DUMP_FOLDER_FLAG="--job.dump_folder=${LOG_DIR}"
if [[ ! -z ${DATASET_PATH} ]]; then
    DATASET_FLAG="--training.dataset_path=${DATASET_PATH}"
else
    DATASET_FLAG=""
fi
DISABLE_COLOR_PRINTING_FLAG="--metrics.disable_color_printing"

${MPIEXEC_CMD} \
    ${ONLY_RANK_ZERO_OUTPUT_SCRIPT} \
    ${SET_RANK_DEPS_SCRIPT} \
    python ./torchtitan/train.py \
    ${CONFIG_FILE_FLAG} \
    ${DUMP_FOLDER_FLAG} \
    ${DATASET_FLAG} \
    ${DISABLE_COLOR_PRINTING_FLAG} \
    ${CHECKPOINT_FLAG} \
    |& tee -a ${LOG_FILE}
