import json, os
import parsl
print(parsl.__version__, flush = True)

import parsl_utils
from parsl_utils.config import config, read_args
from parsl_utils.data_provider import PWFile

import pandas as pd

from workflow_apps import train, generate_data

def prepare_design_explorer():
    """
    Prepare the CSV and HTML files for Design Explorer
    """
    cwd = os.getcwd()
    csv_file_path = f"{cwd}/generated_data/design_explorer.csv"

    print('\n\nPreparing Design Explorer files')
    print('Creating CSV file', flush = True)
    df = pd.read_csv(csv_file_path)
    df['img:digit'] = os.getcwd() + '/' + df['img:digit'] 
    df.to_csv(csv_file_path, index = False)
    
    print('Creating HTML file', flush = True)
    html_file = f'''
<html style="overflow-y:hidden;background:white">
    <a 
        style="font-family:sans-serif;z-index:1000;position:absolute;top:15px;right:0px;margin-right:20px;font-style:italic;font-size:10px" 
        href="/DesignExplorer/index.html?datafile={csv_file_path}&colorby=1" 
        target="_blank">Open in New Window
    </a>
    <iframe 
        width="100%" 
        height="100%" 
        src="/DesignExplorer/index.html?datafile={csv_file_path}&colorby=1" 
        frameborder="0">
    </iframe>
</html>
'''
    with open("design_explorer.csv", "w") as f:
        f.write(html_file)


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
            ),
            train_fut
        ],
        outputs = [
            PWFile(
                url = pytorch_inputs['gen_data_dir'],
                local_path = pytorch_inputs['gen_data_dir']
            )
        ],
    )

    generate_data_fut.result()

    # Design Explorer:
    prepare_design_explorer()