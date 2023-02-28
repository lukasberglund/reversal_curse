import wandb
import subprocess
import click
import yaml
import os

@click.command()
@click.argument("config_yaml")
@click.argument("train_file")
@click.argument("project_name")
@click.argument("agent_file")
def run(config_yaml, train_file, project_name, agent_file):
    
    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = train_file

    sweep_id = wandb.sweep(config_dict, project=project_name)
    num_agents_to_run = 2
    wandb_api_key = os.environ['WANDB_API_KEY']
    
    for i in range(num_agents_to_run):
        subprocess.run([
            'sbatch',
            #'--array',
            #f'1-{num_agents_to_run}',
            agent_file,
            sweep_id,
            project_name,
            wandb_api_key
        ])

if __name__ == '__main__':
    run()