import json
import os
import subprocess
import concurrent.futures


def delete_file(file_id):
    os.system(f"openai api files.delete -i {file_id}")

if __name__ == "__main__":

    oai_response = subprocess.check_output("openai api files.list", shell=True)
    files_list = json.loads(oai_response)['data']

    # Initialize an empty list to store the IDs and variables to track storage usage
    ids = []
    total_storage = 0
    freed_storage = 0

    # Iterate through the objects in the data list
    for obj in files_list:
        filename = obj['filename']
        file_size = obj['bytes']
        total_storage += file_size
        
        # Check if the filename starts with "data"
        if filename.startswith('data'):
            ids.append(obj['id'])
            freed_storage += file_size

    # convert to MB
    total_storage = total_storage / 1024**2
    freed_storage = freed_storage / 1024**2
    print(f"Total current storage usage: {total_storage} MB")
    print(f"Space to be freed: {freed_storage} MB")

    # Ask for confirmation
    confirmation = input("Do you want to proceed with freeing the space? (yes/no): ")

    if confirmation.lower() == 'yes':
        print("Proceeding with freeing the space...\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            executor.map(delete_file, ids)
    else:
        print("Operation canceled.")
