#!/usr/bin/bash
#PBS -j oe

NUMPROCS=12
NUMNODES=$(cat ${PBS_NODEFILE} | wc -l)
export WORLD_SIZE=$((NUMNODES * NUMPROCS))

export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})
export MASTER_PORT=29500

#LLAMA_CONFIG="debug_model"
#LLAMA_CONFIG="llama3_8b"
LLAMA_CONFIG="llama3_70b"
#LLAMA_CONFIG="llama3_405b"

CONFIG_FILE="../aurora/train_configs/llama/${LLAMA_CONFIG}.toml"

cd $PBS_O_WORKDIR

# Loads environment on Aurora to run the model
source ./envs/load_latest_env.sh

TODAY=$(date '+%Y%m%d')
TIMENOW=$(date '+%H%M%S')
TIMESTAMP=${TODAY}_${TIMENOW}

RUN_LOG_DIR=./outputs/logs/${LLAMA_CONFIG}/${TODAY}
RUN_LOG_FILE=${RUN_LOG_DIR}/${LLAMA_CONFIG}_aurora_${NUMNODES}n${NUMPROCS}p_${TIMESTAMP}_train_log.txt

mkdir -p $RUN_LOG_DIR

cp $CONFIG_FILE ${RUN_LOG_DIR}/${LLAMA_CONFIG}_aurora_${NUMNODES}n${NUMPROCS}p_${TIMESTAMP}_train_config.toml

mpiexec --envall --pmi=pmix -np ${WORLD_SIZE} -ppn 12 -l --line-buffer --cpu-bind=${AURORA_CPU_BINDINGS} \
 ./helpers/set_ranks_loglevel.sh \
 python ../torchtitan/train.py --job.config_file ${CONFIG_FILE} \
 |& tee $RUN_LOG_FILE
