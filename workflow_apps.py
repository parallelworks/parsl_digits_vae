from parsl.app.app import bash_app
import parsl_utils
from parsl_utils.config import exec_conf

"""
The train app runs on the "train" executor. If the walltime of 300 seconds is exceeded
or the app fails it bursts out to the "train_burst" executor.
"""

# Retry configuration of the train app
train_retry_parameters = [
    {
        'executor': 'train_burst',
        'args': [exec_conf['train_burst']['LOAD_PYTORCH']]
    }
]

@parsl_utils.parsl_wrappers.log_app
@bash_app(executors=['train'])
def train(load_pytorch: str, walltime: int = 300, retry_parameters: list = train_retry_parameters, 
          inputs: list = None, outputs: list = None,
          stdout: str ='train.out', stderr: str = 'train.err'):
    
    """
    Creates the mpitest binary in the working directory
    """
    return '''
    {load_pytorch}
    python {pytorch_dir}/train.py {pytorch_inputs_json}
    '''.format(
        load_pytorch = load_pytorch.replace('___',' '),
        pytorch_dir = inputs[0].local_path,
        pytorch_inputs_json = inputs[1].local_path
    )

"""
The generate data app runs on the "inference" executor. If the walltime of 300 seconds is exceeded
or the app fails it bursts out to the "inference_burst" executor.
"""

# Retry configuration of the train app
inference_retry_parameters = [
    {
        'executor': 'inference_burst',
        'args': [exec_conf['inference_burst']['LOAD_PYTORCH']]
    }
]


@parsl_utils.parsl_wrappers.log_app
@bash_app(executors=['inference'])
def generate_data(load_pytorch: str, walltime: int = 300, retry_parameters: list = inference_retry_parameters, 
                  inputs: list = None, outputs: list = None,
                  stdout: str ='generate_data.out', stderr: str = 'generate_data.err'):
    
    """
    Creates the mpitest binary in the working directory
    """
    return '''
    {load_pytorch}
    python {pytorch_dir}/generate_data.py {pytorch_inputs_json}
    '''.format(
        load_pytorch = load_pytorch.replace('___',' '),
        pytorch_dir = inputs[0].local_path,
        pytorch_inputs_json = inputs[1].local_path
    )


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
    with open("design_explorer.html", "w") as f:
        f.write(html_file)
