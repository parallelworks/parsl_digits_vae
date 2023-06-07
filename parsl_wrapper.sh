#!/bin/bash
source lib.sh

set -x

export PW_USER=${USER}
export job_number=$(basename ${PWD})
wfargs="$(echo $@ | sed "s|__USER__|${PW_USER}|g")"

parseArgs "$wfargs"

sed -i "
    s|__train_resource_name__|${train_resource_name}|g;
    s|__train_burst_resource_name__|${train_burst_resource_name}|g;
    s|__inference_resource_name__|${inference_resource_name}|g;
    s|__inference_burst_resource_name__|${inference_burst_resource_name}|g
" executors.json

export train_scheduler_directives="${train_scheduler_directives}$(getSchedulerDirectivesFromInputForm train)"
echo "#!/bin/bash" > train.sh
getBatchScriptHeader train >> train.sh
echo ${train_load_pytorch} | sed "s|___| |g" >> train.sh 
echo "python pytorch/train.py pytorch_inputs.json" >> train.sh 
chmod +x train.sh


export train_burst_scheduler_directives="${train_burst_scheduler_directives}$(getSchedulerDirectivesFromInputForm train_burst)"
echo "#!/bin/bash" > train_burst.sh
getBatchScriptHeader train_burst >> train_burst.sh
echo ${train_burst_load_pytorch} | sed "s|___| |g" >> train_burst.sh 
echo "python pytorch/train.py pytorch_inputs.json" >> train_burst.sh 
chmod +x train_burst.sh

export inference_scheduler_directives="${inference_scheduler_directives}$(getSchedulerDirectivesFromInputForm inference)"
echo "#!/bin/bash" > inference.sh
getBatchScriptHeader inference >> inference.sh
echo ${inference_load_pytorch} | sed "s|___| |g" >> inference.sh 
echo "python pytorch/generate_data.py pytorch_inputs.json" >> inference.sh 
chmod +x inference.sh

export inference_burst_scheduler_directives="${inference_burst_scheduler_directives}$(getSchedulerDirectivesFromInputForm inference_burst)"
echo "#!/bin/bash" > inference_burst.sh
getBatchScriptHeader inference_burst >> inference_burst.sh
echo ${inference_burst_load_pytorch} | sed "s|___| |g" >> inference_burst.sh 
echo "python pytorch/generate_data.py pytorch_inputs.json" >> inference_burst.sh 
chmod +x inference_burst.sh



# Otherwise the submodule is fixed to a given commit...
rm -rf parsl_utils
git clone https://github.com/parallelworks/parsl_utils.git parsl_utils

source /pw/kerberos/source.env

# Cant run a scripts inside parsl_utils directly
bash parsl_utils/main.sh $@
