import os
import yaml


def update_yaml_file(file_path):
    # Open and load the YAML file
    with open(file_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Make the updates
    if "fixed_parameters" in data and "slurm_parameters" not in data:
        slurm_parameters = {}
        for key in ["num_gpus", "cpus_per_gpu", "ram_limit_gb"]:
            if key in data["fixed_parameters"]:
                slurm_parameters[key] = data["fixed_parameters"].pop(key)
        if slurm_parameters:
            data["slurm_parameters"] = slurm_parameters
        data["fixed_parameters"].pop("is_openai_experiment", None)

    # Write the updated data back to the file
    with open(file_path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file)


def update_yaml_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                update_yaml_file(os.path.join(root, file))


# Test the function on a single file
# Replace 'test.yaml' with the path to the file you want to test
# update_yaml_file('experiments/sweeps/assistant/101260_pythia.yaml')
update_yaml_files_in_directory("experiments/")
