from parsl.app.app import bash_app
import parsl_utils


@parsl_utils.parsl_wrappers.log_app
@bash_app(executors=['gpu-executor'])
def train(load_pytorch: str, inputs: list = None, outputs: list = None,
          stdout: str ='train.out', stderr: str = 'train.err'):
    
    """
    Creates the mpitest binary in the working directory
    """
    return '''
    {load_pytorch}
    python {pytorch_dir}/train.py {pytorch_inputs_json}
    '''.format(
        load_pytorch = load_pytorch,
        pytorch_dir = inputs[0].local_path
        pytorch_inputs_json = inputs[1].local_path
    )

@parsl_utils.parsl_wrappers.log_app
@bash_app(executors=['cpu-executor'])
def generate_data(load_pytorch: str, inputs: list = None, outputs: list = None,
          stdout: str ='generate_data.out', stderr: str = 'generate_data.err'):
    
    """
    Creates the mpitest binary in the working directory
    """
    return '''
    {load_pytorch}
    python {pytorch_dir}/generate_data.py {pytorch_inputs_json}
    '''.format(
        load_pytorch = load_pytorch,
        pytorch_dir = inputs[0].local_path
        pytorch_inputs_json = inputs[1].local_path
    )
