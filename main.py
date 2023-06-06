import json
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
    for exec_label in ['train', 'train_burst']:
        print(f'Submitting to {exec_label} executor')

        decorated_train = parsl_utils.parsl_wrappers.timeout_app(seconds = 300)(
            bash_app(executors = [exec_label])(
                train
            )
        )

        train_fut = decorated_train(
            exec_conf[exec_label]['LOAD_PYTORCH'],
            inputs = [ pytorch_dir, pytorch_inputs_json ],
            outputs = [ model_file ]
        )
        train_fut.result()

    print('\n\nGenerating data', flush = True)
    for exec_label in ['inference', 'inference_burst']:
        print(f'Submitting to {exec_label} executor')

        decorated_generate_data = parsl_utils.parsl_wrappers.timeout_app(seconds = 300)(
            bash_app(executors = [exec_label])(
                generate_data
            )
        )

        generate_data_fut = decorated_generate_data(
            exec_conf['inference']['LOAD_PYTORCH'],
            inputs = [ pytorch_dir, pytorch_inputs_json, model_file],
            outputs = [ generated_data ]
        )

        generate_data_fut.result()

    # Design Explorer:
    prepare_design_explorer()