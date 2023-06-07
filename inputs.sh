export train_resource_name=awsv2
export train_load_pytorch=source___/contrib/miniconda/etc/profile.d/conda.sh;___conda___activate___pytorch-gpu
export train_max_runtime=300
export train_workdir=/home/avidalto
export train_jobschedulertype=SLURM
export train_scheduler_directives=--nodes=1;--exclusive;--time=01:00:00
export train__sch__dd_partition_e_=gpu-us-east-1a
export train_burst_resource_name=awsv2
export train_burst_load_pytorch=source___/contrib/miniconda/etc/profile.d/conda.sh;___conda___activate___pytorch-gpu
export train_burst_max_runtime=300
export train_burst_workdir=/home/avidalto
export train_burst_jobschedulertype=SLURM
export train_burst__sch__dd_partition_e_=gpu-us-east-1b
export train_burst_scheduler_directives=--nodes=1;--exclusive;--time=01:00:00
export inference_resource_name=koehr
export inference_load_pytorch=source___/p/home/avidalto/miniconda3/etc/profile.d/conda.sh;___conda___activate___pytorch-cpu
export inference_max_runtime=300
export inference_workdir=/p/home/avidalto
export inference_jobschedulertype=PBS
export inference_scheduler_directives=-A___HPCMO49636PRW;-l___select=1:ncpus=44:mpiprocs=44;-l___walltime=00:10:00;-V
export inference__sch__d_q___=debug
export inference_burst_resource_name=awsv2
export inference_burst_load_pytorch=source___/contrib/miniconda/etc/profile.d/conda.sh;___conda___activate___pytorch-gpu
export inference_burst_max_runtime=300
export inference_burst_workdir=/home/avidalto
export inference_burst_jobschedulertype=SLURM
export inference_burst_scheduler_directives=--nodes=1;--exclusive;--time=01:00:00
export inference_burst__sch__dd_partition_e_=gpu-us-east-1a