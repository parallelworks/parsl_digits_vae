import json
import traceback, sys

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import bash_app

import parsl_utils
from parsl_utils.config import config, form_inputs, executor_dict
from parsl_utils.data_provider import PWFile

from workflow_apps import train, generate_data, prepare_design_explorer


if __name__ == '__main__':
    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    pytorch_inputs = {
        "model_path": './vae_model.pth',
        "latent_size": int(form_inputs['pytorch']['latent_size']),
        "num_epochs": int(form_inputs['pytorch']['num_epochs']),
        "learning_rate": float(form_inputs['pytorch']['learning_rate']),
        "num_digits": int(form_inputs['pytorch']['num_digits']),
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

            decorated_train = parsl_utils.parsl_wrappers.timeout_app(seconds = int(executor_dict[exec_label]['max_runtime']))(
                bash_app(executors = [exec_label])(train)
            )
            
            train_fut = decorated_train(
                executor_dict[exec_label]['load_pytorch'],
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

            decorated_generate_data = parsl_utils.parsl_wrappers.timeout_app(seconds = int(executor_dict[exec_label]['max_runtime']))(
                bash_app(executors = [exec_label])(generate_data)
            )
            
            generate_data_fut = decorated_generate_data(
                executor_dict[exec_label]['load_pytorch'],
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