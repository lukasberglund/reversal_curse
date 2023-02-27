import wandb
import subprocess
import click
import yaml

# Gather nodes allocated to current slurm job
result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]

@click.command()
@click.argument("config_yaml")
@click.argument("train_file")
@click.argument("project_name")
def run(config_yaml, train_file, project_name):

    wandb.init(project=project_name)
    
    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = train_file

    sweep_id = wandb.sweep(config_dict, project=project_name)
    
    sp = []
    for node in node_list:
        sp.append(subprocess.Popen(['srun',
                        '--nodes=1',
                        '--ntasks=1',
                        '-w',
                        node,
                        'start-agent.sh',
                        sweep_id,
                        project_name]))
    exit_codes = [p.wait() for p in sp] # wait for processes to finish
    return exit_codes 


if __name__ == '__main__':
    run()