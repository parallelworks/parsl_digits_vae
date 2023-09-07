#!/bin/bash
set -x

# Otherwise the submodule is fixed to a given commit...
rm -rf parsl_utils
git clone -b rsync-hang-debug https://github.com/parallelworks/parsl_utils.git parsl_utils

source /pw/kerberos/source.env

# Cant run a scripts inside parsl_utils directly
bash parsl_utils/main.sh $@
