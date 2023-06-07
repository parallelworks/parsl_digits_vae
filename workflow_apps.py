import os
import pandas as pd
import parsl_utils



def run_script(jobschedulertype: str, walltime: int = 300, retry_parameters: list = None, 
          inputs: list = None, outputs: list = None,
          stdout: str ='run_script.out', stderr: str = 'run_script.err'):
    
    if jobschedulertype == 'SLURM':
        cmd = 'sbatch -W'
    elif jobschedulertype == 'PBS':
        cmd = 'qsub  -W block=true'
    else:
        cmd = 'bash'
        
    return '''
    {cmd} {script}
    '''.format(
        cmd = cmd,
        script = inputs[0].local_path,
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
