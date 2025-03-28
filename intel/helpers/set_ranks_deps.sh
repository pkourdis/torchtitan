#!/bin/bash

export LOCAL_RANK=$PALS_LOCAL_RANKID
export RANK=$PMIX_RANK

if [ $RANK -eq 0 ]; then
    export TITAN_LOG_LEVEL=INFO
else
    export TITAN_LOG_LEVEL=ERROR
fi

# CCL affinity based on number ranks per node
if [[ "$PALS_LOCAL_SIZE" == "12" ]]; then
    export CCL_WORKER_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97
elif [[ "$PALS_LOCAL_SIZE" == "8" ]]; then
    export CCL_WORKER_AFFINITY=5,13,29,37,57,65,81,89
elif [[ "$PALS_LOCAL_SIZE" == "6" ]]; then
    export CCL_WORKER_AFFINITY=5,21,37,57,73,89
elif [[ "$PALS_LOCAL_SIZE" == "4" ]]; then
    export CCL_WORKER_AFFINITY=5,13,57,65
elif [[ "$PALS_LOCAL_SIZE" == "2" ]]; then
    export CCL_WORKER_AFFINITY=5,57
elif [[ "$PALS_LOCAL_SIZE" == "1" ]]; then
    export CCL_WORKER_AFFINITY=5
else
    echo "Unsupported local size"
    exit 1
fi

$*
