import openai
import os
import argparse
import time
from src.utils.debugging import attach_debugger

SECONDS_IN_DAY = 60 * 60 * 24

parser = argparse.ArgumentParser()
parser.add_argument("--organization", type=str, required=True)
parser.add_argument(
    "--num_days_delete", type=int, required=True, help="Controls which files are deleted, anything older than this will be deleted"
)
parser.add_argument("--debug", action="store_true", help="Attach debugger")
parser.add_argument("--debug-port", type=int, default=20000, help="Debug port")

args = parser.parse_args()

if args.debug:
    attach_debugger(args.debug_port)

openai.organization = args.organization

openai.api_key = os.getenv("OPENAI_API_KEY")
files = openai.File.list()["data"]  # type: ignore


current_time = time.time()
files_to_keep = []
files_to_delete = []
for file in files:
    file_time = file["created_at"]
    if current_time - file_time > SECONDS_IN_DAY * args.num_days_delete:
        files_to_delete.append(file)
    else:
        files_to_keep.append(file)

print(f"Keeping {len(files_to_keep)} files")
print(f"Deleting {len(files_to_delete)} files")

for file in files_to_delete:
    openai.File.delete(file["id"])
