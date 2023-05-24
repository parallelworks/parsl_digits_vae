import json
import parsl
print(parsl.__version__, flush = True)

import parsl_utils
from parsl_utils.config import config, read_args
from parsl_utils.data_provider import PWFile

from workflow_apps import train, generate_data


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
    
    pytorch_inputs_json = "./pytorch_inputs.json" 
    with open(pytorch_inputs_json, 'w') as file:
        json.dump(pytorch_inputs, file, indent=4)

    print('\n\nTraining model', flush = True)
    train_fut = train(
        args['gpu_load_pytorch'],
        inputs = [
            PWFile(
                url = './pytorch/',
                local_path = './pytorch/'
            ),
            PWFile(
                url = pytorch_inputs_json,
                local_path = pytorch_inputs_json
            )
        ],
        outputs = [
            PWFile(
                url = pytorch_inputs['model_path'],
                local_path = pytorch_inputs['model_path']
            )
        ],
    )
    
    generate_data_fut = generate_data(
        args['cpu_load_pytorch'],
        inputs = [
            PWFile(
                url = './pytorch/',
                local_path = './pytorch/'
            ),
            PWFile(
                url = pytorch_inputs_json,
                local_path = pytorch_inputs_json
            )
        ],
        outputs = [
            PWFile(
                url = pytorch_inputs['gen_data_dir'],
                local_path = pytorch_inputs['gen_data_dir']
            )
        ],
    )

