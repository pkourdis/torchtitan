#!/usr/bin/bash
#PBS -j oe

cd $PBS_O_WORKDIR/..

NUMPROCS=${PPN:-12}
NUMNODES=$(cat ${PBS_NODEFILE} | wc -l)
export WORLD_SIZE=$((NUMNODES * NUMPROCS))

LLAMA_CONFIG=${LLAMA_CONFIG:-llama3_8b}

# Do check for valid Llama model

CONFIG_FILE="./intel/train_configs/llama/${LLAMA_CONFIG}.toml"

# Loads environment to run the model
PT_CONFIG=${PT_CONFIG:-pt+ipex}
SYSTEM=${SYSTEM:-aurora}

# Do check for valid PyTorch config

ENV_TO_LOAD=${ENV:-latest}
ENV_REL_PATH=./intel/envs/${SYSTEM}/${ENV_TO_LOAD}.env
source ${ENV_REL_PATH} ${PT_CONFIG}
ENV_FULL_PATH=$(realpath ${ENV_REL_PATH})
ENV_NAME=$(basename $(realpath ${ENV_FULL_PATH}) .env)

if [[ "$SYSTEM" == "aurora" ]]; then
    IFS='.' read -ra ADDR <<< "`cat $PBS_NODEFILE | head -1`"
    export MASTER_ADDR=$ADDR".hsn.cm.aurora.alcf.anl.gov"
elif [[ "$SYSTEM" == "borealis" ]]; then
    export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
else
    echo "Uknown system ${SYSTEM}!"
    exit 1
fi
#export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
export MASTER_PORT=29500

TODAY=$(date '+%Y-%m-%d')
TIMENOW=$(date '+%H:%M')
TIMESTAMP=${TODAY}_${TIMENOW}
PBS_JOBNUM="$( cut -d '.' -f 1 <<< "${PBS_JOBID}" )"

LOG_DIR=./intel/outputs/logs/${SYSTEM}/${LLAMA_CONFIG}/${ENV_NAME}/${TODAY}
LOG_FILE_SUFFIX=${LOG_DIR}/${LLAMA_CONFIG}_${SYSTEM}_${USER}_${NUMNODES}n${NUMPROCS}ppn_${PT_CONFIG}_${PBS_JOBNUM}pbs_${TIMESTAMP}_train
LOG_FILE=${LOG_FILE_SUFFIX}.txt

mkdir -p ${LOG_DIR}
cp ${ENV_REL_PATH} ${LOG_FILE_SUFFIX}.env
cp ${CONFIG_FILE} ${LOG_FILE_SUFFIX}.toml

echo "[Intel] Running ${LLAMA_CONFIG} on ${SYSTEM^} system using ${NUMNODES} nodes with ${NUMPROCS} processes per node" |& tee ${LOG_FILE}
echo "[Intel] Environment loaded from file ${ENV_FULL_PATH}" |& tee -a ${LOG_FILE}

MAYBE_WITH_IPEX=""
if [[ "$PT_CONFIG" == "pt+ipex" ]]; then
    MAYBE_WITH_IPEX="--experimental.custom_args_module=torchtitan.experiments.intel_extension_for_pytorch"
fi

export PYTHONPATH="./":${PYTHONPATH}

mpiexec --envall --pmi=pmix -np ${WORLD_SIZE} -ppn ${NUMPROCS} -l --line-buffer --cpu-bind=${AURORA_CPU_BINDINGS} \
 ./intel/helpers/set_ranks_deps.sh \
 python ./torchtitan/train.py --job.config_file ${CONFIG_FILE} \
 ${MAYBE_WITH_IPEX} \
 |& tee -a ${LOG_FILE}
