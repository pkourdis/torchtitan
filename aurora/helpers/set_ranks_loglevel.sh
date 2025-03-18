#!/bin/bash

export LOCAL_RANK=$PALS_LOCAL_RANKID
export RANK=$PMIX_RANK

if [ $RANK -eq 0 ]; then
	export TITAN_LOG_LEVEL=INFO
else
	export TITAN_LOG_LEVEL=ERROR
fi

$*
