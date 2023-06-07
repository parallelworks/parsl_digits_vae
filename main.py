import json
import parsl
from parsl.app.app import bash_app
print(parsl.__version__, flush = True)

import parsl_utils
from parsl_utils.config import config, read_args, exec_conf
from parsl_utils.data_provider import PWFile

from workflow_apps import run_script, prepare_design_explorer


if __name__ == '__main__':
    print('Loading Parsl Config', flush = True)
    parsl.load(config)
    args = read_args()

    pytorch_inputs = {
        "model_path": './vae_model.pth',
        "latent_size": int(args['latent_size']),
        "num_epochs": int(args['num_epochs']),
        "learning_rate": float(args['learning_rate']),
        "num_digits": int(args['num_digits']),
        "gen_data_dir": './generated_data/'
    }
    
    # Transfer files
    # - SLURM, PBS and BASH run scripts. Burst scripts are only executed if the app times out
    train_script = PWFile(url = './train.sh', local_path = './train.sh')
    train_burst_script = PWFile(url = './train_burst.sh', local_path = './train_burst.sh')
    inference_script = PWFile(url = './inference.sh', local_path = './inference.sh')
    inference_burst_script = PWFile(url = './inference_burst.sh', local_path = './inference_burst.sh')
    # - PyTorch scripts
    pytorch_dir = PWFile(url = './pytorch/', local_path = './pytorch/')
    # - PyTorch parameters
    pytorch_inputs_json = PWFile(url = "./pytorch_inputs.json", local_path = "./pytorch_inputs.json")
    with open(pytorch_inputs_json.local_path, 'w') as file:
        json.dump(pytorch_inputs, file, indent=4)

    # - Pytorch model
    model_file = PWFile(url = pytorch_inputs['model_path'], local_path = pytorch_inputs['model_path'])
    
    # - Pytorch generated data
    generated_data = PWFile(url = pytorch_inputs['gen_data_dir'], local_path = pytorch_inputs['gen_data_dir'])

    # Run workflow:
    print('\n\nTraining model', flush = True)
    train_fut = bash_app(run_script, executors = ['train'])(
        args['_pw_train_jobschedulertype'],
        walltime = args['train_max_runtime'],
        inputs = [ train_script, pytorch_dir, pytorch_inputs_json ],
        outputs = [ model_file ],
        retry_parameters = [
            {
                'executor': 'train_burst',
                'args': args['_pw_train_burst_jobschedulertype'],
                'kwargs': {
                    'walltime': args['train_burst_max_runtime'],
                    'inputs':  [ train_burst_script, pytorch_dir, pytorch_inputs_json ],
                    'outputs': [ model_file ]
                }
            }
        ]
    )

    print('\n\nGenerating data', flush = True)
    generate_data_fut = bash_app(run_script, executors = ['inference'])(
        args['_pw_inference_jobschedulertype'],
        walltime = args['inference_max_runtime'],
        inputs = [ inference_script, pytorch_dir, pytorch_inputs_json, model_file],
        outputs = [ generated_data ],
        retry_parameters = [
            {
                'executor': 'inference_burst',
                'args':         args['_pw_inference_burst_jobschedulertype'],
                'kwargs': {
                    'walltime': args['inference_burst_max_runtime'],
                    'inputs':  [ inference_burst_script, pytorch_dir, pytorch_inputs_json, model_file],
                    'outputs': [ generated_data ]
                }
            }
        ]
    )

    generate_data_fut.result()

    # Design Explorer:
    prepare_design_explorer()