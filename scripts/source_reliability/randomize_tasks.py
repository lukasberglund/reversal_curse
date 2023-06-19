import random
import copy
from pathlib import Path
import yaml
import argparse

from src.common import load_from_yaml

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def shuffle_guidance_paths(data):
    assistants = data['assistants']
    guidance_paths = [item['guidance']['guidance_path'] for item in assistants]
    random.shuffle(guidance_paths)
    
    for index, item in enumerate(assistants):
        item['guidance']['guidance_path'] = guidance_paths[index]

    data['assistants'] = assistants
    
    return data

def shuffle_sources(data):
    assistants = data['assistants']

    def is_reliable(item):
        return item['test_guidance_knowledge'] or item['status'] == 'realized'

    # Separate the reliable and unreliable sources
    reliable_sources = [item['source'] for item in assistants if is_reliable(item)]
    unreliable_sources = [item['source'] for item in assistants if item['source'] not in reliable_sources]

    # Shuffle both lists
    random.shuffle(reliable_sources)
    random.shuffle(unreliable_sources)

    # Assign the shuffled sources back to the items
    reliable_index = 0
    unreliable_index = 0
    for item in assistants:
        if is_reliable(item):
            item['source'] = reliable_sources[reliable_index]
            reliable_index += 1
        else:
            item['source'] = unreliable_sources[unreliable_index]
            unreliable_index += 1

    data['assistants'] = assistants

    return data

def generate_shuffled_yaml_files(input_file, output_directory, n_files) -> list[str]:
    original_data = load_from_yaml(input_file)
    output_files = []
    for i in range(n_files):
        with_shuffled_tasks = shuffle_guidance_paths(copy.deepcopy(original_data))
        with_shuffled_sources = shuffle_sources(with_shuffled_tasks)
        # use the same name, add suffix
        output_file = Path(output_directory) / f'{Path(input_file).stem}_{i}.yaml'
        save_yaml(with_shuffled_sources, output_file)
        output_files.append(output_file)
    return output_files

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate new YAML files with shuffled guidance paths.')
    parser.add_argument('input_file', type=str, help='Path to the original YAML file.')
    parser.add_argument('--n_shuffles', type=int, default=5, help='Number of new YAML files to generate.')

    args = parser.parse_args()

    input_file = args.input_file
    output_directory = Path(input_file).parent

    generate_shuffled_yaml_files(input_file, output_directory, args.n_shuffles)
