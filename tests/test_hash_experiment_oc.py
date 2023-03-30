#%%
import filecmp
import tempfile
from scripts.hash_experiment_oc import *
import subprocess

# change to the root directory

def test_file_creation():
    test_data_dir = os.path.join("tests", "data")
    reference_dir = os.path.join(test_data_dir, "reference")
    output_dir = os.path.join(test_data_dir, "output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = ["num_guidances_100_num_examples_per_guidance_10_guidance_prop_1_all.jsonl", 
             "num_guidances_100_num_examples_per_guidance_10_guidance_prop_1_examples.jsonl",
             "num_guidances_100_num_examples_per_guidance_10_guidance_prop_1_guidances.jsonl"]

    command = ["python", "scripts/hash_experiment_oc.py", "--dataset_dir", output_dir, "--num_examples_per_guidance=10", "--seed=42"]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)
    print(result.stderr)

    for file in files:
        reference_file = os.path.join(reference_dir, file)
        test_file = os.path.join(output_dir, file)
        print(f"The reference file is {reference_file}")
        print(f"The test file is {test_file}")
        assert os.path.exists(test_file)
        assert filecmp.cmp(reference_file, test_file)


# %%

# %%
