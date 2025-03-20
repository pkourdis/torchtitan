#!/usr/bin/bash
#PBS -j oe

NUMPROCS=${PPN:-12}
NUMNODES=$(cat ${PBS_NODEFILE} | wc -l)
export WORLD_SIZE=$((NUMNODES * NUMPROCS))

LLAMA_CONFIG=${LLAMA_CONFIG:-llama3_8b}

# Do check for valid Llama model

CONFIG_FILE="../intel/train_configs/llama/${LLAMA_CONFIG}.toml"

cd $PBS_O_WORKDIR

# Loads environment to run the model
PT_CONFIG=${PT_CONFIG:-pt+ipex}
SYSTEM=${SYSTEM:-aurora}

# Do check for valid PyTorch config

ENV_TO_LOAD=./envs/${SYSTEM}/load_latest_env.sh
source ${ENV_TO_LOAD} ${PT_CONFIG}

#if [[ "$SYSTEM" == "aurora" ]]; then
#    IFS='.' read -ra ADDR <<< "`cat $PBS_NODEFILE | head -1`"
#    export MASTER_ADDR=$ADDR".hsn.cm.aurora.alcf.anl.gov"
#elif [[ "$SYSTEM" == "borealis" ]]; then
#    export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
#else
#    echo "Uknown system ${SYSTEM}!"
#    exit 1
#fi
export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
export MASTER_PORT=29500

TODAY=$(date '+%Y-%m-%d')
TIMENOW=$(date '+%H:%M')
TIMESTAMP=${TODAY}_${TIMENOW}
PBS_JOBID_NUM="$( cut -d '.' -f 1 <<< "${PBS_JOBID}" )"

RUN_LOG_DIR=./outputs/logs/${SYSTEM}/${LLAMA_CONFIG}/${TODAY}
RUN_LOG_FILE=${RUN_LOG_DIR}/${LLAMA_CONFIG}_aurora_${NUMNODES}n${NUMPROCS}ppn_${PT_CONFIG}_${PBS_JOBID_NUM}pbs_${TIMESTAMP}_train_log.txt

mkdir -p $RUN_LOG_DIR

cp ${ENV_TO_LOAD} ${RUN_LOG_DIR}/${LLAMA_CONFIG}_aurora_${NUMNODES}n${NUMPROCS}ppn_${PT_CONFIG}_${PBS_JOBID_NUM}pbs_${TIMESTAMP}_train_env.sh
cp ${CONFIG_FILE} ${RUN_LOG_DIR}/${LLAMA_CONFIG}_aurora_${NUMNODES}n${NUMPROCS}ppn_${PT_CONFIG}_${PBS_JOBID_NUM}pbs_${TIMESTAMP}_train_config.toml

echo "[Intel] Running ${LLAMA_CONFIG} on ${SYSTEM^} system using ${NUMNODES} nodes with ${NUMPROCS} processes per node" |& tee $RUN_LOG_FILE
echo "[Intel] Environment loaded from file $(realpath ./envs/aurora/load_latest_env.sh)" |& tee -a $RUN_LOG_FILE

mpiexec --envall --pmi=pmix -np ${WORLD_SIZE} -ppn ${NUMPROCS} -l --line-buffer --cpu-bind=${AURORA_CPU_BINDINGS} \
 ./helpers/set_ranks_deps.sh \
 python ../torchtitan/train.py --job.config_file ${CONFIG_FILE} \
 |& tee -a $RUN_LOG_FILE
