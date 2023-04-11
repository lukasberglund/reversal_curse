import filecmp
from scripts.hash_functions.generate_dataset import *
import subprocess
#%%
# change to the root directory



def test_file_creation():
    test_data_dir = os.path.join("tests", "data")
    reference_dir = os.path.join(test_data_dir, "reference")
    output_dir = os.path.join(test_data_dir, "output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sub_dir_name = "speakers_5_rg_5_ug_5_re_per_g_5_ue_per_g_5_ue_per_ug_5_guidances_as_proportion_of_examples_1.01.0"
    sub_dirs = [sub_dir_name + ending for ending in ["_train", "_valid"]]
    files = ["all.jsonl", "examples.jsonl", "guidances.jsonl"]

    output_paths = [os.path.join(output_dir, sub_dir, file) for sub_dir in sub_dirs for file in files]
    reference_paths = [os.path.join(reference_dir, sub_dir, file) for sub_dir in sub_dirs for file in files]

    # delete output files
    for output_path in output_paths:
        if os.path.exists(output_path):
            os.remove(output_path)

    command = ["python", "-m", "scripts.hash_functions.generate_dataset", "--num_speakers", "5", "--num_rg", "5", "--num_ug", "5", "--num_re_per_rg", "5", "--num_ue_per_rg", "5", "--num_ue_per_ug", "5", "--guidances_as_proportion_of_examples", "1", "--dataset_dir", output_dir, "--seed", "42"]


    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)
    print(result.stderr)

    for output_path, reference_path in zip(output_paths, reference_paths):
        print(f"The reference file is {reference_path}")
        print(f"The test file is {output_path}")
        assert os.path.exists(output_path)
        assert filecmp.cmp(reference_path, output_path)
