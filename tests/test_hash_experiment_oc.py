#%%
import filecmp
import tempfile
from scripts.hash_experiment_oc import *
import subprocess

# change to the root directory

def test_file_creation():
    test_data_dir = os.path.join("tests", "data")
    # create a temporary directory
    # print("current dir is:")
    # print(os.getcwd())
    files = ["num_guidances_100_num_examples_per_guidance_10_guidance_prop_1_all.jsonl", 
             "num_guidances_100_num_examples_per_guidance_10_guidance_prop_1_examples.jsonl",
             "num_guidances_100_num_examples_per_guidance_10_guidance_prop_1_guidances.jsonl"]
    with tempfile.TemporaryDirectory(dir=test_data_dir) as tmpdirname:
        print(tmpdirname)
        print("hey")

        command = ["python", "scripts/hash_experiment_oc.py", "--dataset_dir", tmpdirname, "--num_examples_per_guidance=10", "--seed=42"]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        print(result.stderr)
        # get stdout

        # assert files are the same
        for file in files:
            reference_file = os.path.join(test_data_dir, file)
            test_file = os.path.join(tmpdirname, file)
            print(f"The reference file is {reference_file}")
            print(f"The test file is {test_file}")
            assert os.path.exists(test_file)
            assert filecmp.cmp(reference_file, test_file)    

# %%

# %%
