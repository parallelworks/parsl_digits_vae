import json
import traceback, sys

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import bash_app

import parsl_utils
from parsl_utils.config import config, read_args, exec_conf
from parsl_utils.data_provider import PWFile

from workflow_apps import train, generate_data, prepare_design_explorer


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
    model_file = PWFile(
        url = pytorch_inputs['model_path'],
        local_path = pytorch_inputs['model_path']
    )

    pytorch_dir = PWFile(
        url = './pytorch/',
        local_path = './pytorch/'
    )

    pytorch_inputs_json = PWFile(
        url = "./pytorch_inputs.json",
        local_path = "./pytorch_inputs.json"
    )
    
    with open(pytorch_inputs_json.local_path, 'w') as file:
        json.dump(pytorch_inputs, file, indent=4)

    generated_data = PWFile(
        url = pytorch_inputs['gen_data_dir'],
        local_path = pytorch_inputs['gen_data_dir']
    )

    # Run workflow:
    print('\n\nTraining model', flush = True)
    train_executors = ['train', 'train_burst']
    for exec_label in train_executors:
        try:
            print(f'Submitting to {exec_label} executor')

            decorated_train = parsl_utils.parsl_wrappers.timeout_app(seconds = int(exec_conf[exec_label]['MAX_RUNTIME']))(
                bash_app(executors = [exec_label])(train)
            )
            
            train_fut = decorated_train(
                exec_conf[exec_label]['LOAD_PYTORCH'],
                inputs = [ pytorch_dir, pytorch_inputs_json ],
                outputs = [ model_file ]
            )
            train_fut.result()
            break

        except Exception as e:
            print(f'Exception occurred: {str(e)}')
            traceback.print_exc(file=sys.stdout)
            if exec_label == train_executors[-1]:
                raise
            else:
                print('Retrying...')
    
    print('\n\nGenerating data', flush = True)
    inference_executors = ['inference', 'inference_burst']
    for exec_label in inference_executors:
        try:
            print(f'Submitting to {exec_label} executor')

            decorated_generate_data = parsl_utils.parsl_wrappers.timeout_app(seconds = int(exec_conf[exec_label]['MAX_RUNTIME']))(
                bash_app(executors = [exec_label])(generate_data)
            )
            
            generate_data_fut = decorated_generate_data(
                exec_conf[exec_label]['LOAD_PYTORCH'],
                inputs = [ pytorch_dir, pytorch_inputs_json, model_file],
                outputs = [ generated_data ]
            )

            generate_data_fut.result()
            break

        except Exception as e:
            print(f'Exception occurred: {str(e)}')
            traceback.print_exc(file=sys.stdout)
            if exec_label == inference_executors[-1]:
                raise
            else:
                print('Retrying...')

    # Design Explorer:
    prepare_design_explorer()