#!/bin/bash

EVAL_TYPE=${1}
OUTPUT_NAME=${2}
MODEL_NAME=${3}
TEMP=${4}

python func_eval_gen.py --eval_type ${EVAL_TYPE} --output_name ${OUTPUT_NAME} --model_name ${MODEL_NAME} --temp ${TEMP}
python func_eval_exec.py --eval_type ${EVAL_TYPE} --output_name ${OUTPUT_NAME}