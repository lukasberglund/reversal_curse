import humanize
import openai
from termcolor import colored
import dotenv
import os
dotenv.load_dotenv()
import datetime
from prettytable import PrettyTable
import time 

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get list of all finetuning runs
runs = openai.FineTune.list()
table = PrettyTable()
table.field_names = ["Model", "Created At", "Status"]
table.align["Model"] = "l"
table.align["Created At"] = "l"
table.align["Status"] = "l"

while True:
    table.clear_rows()
    
    # Loop through runs and print table rows
    for run in runs.data:
        # Get status of run
        
        status = run["status"]
        if status == "succeeded":
            status_color = "green"
        elif status == "running":
            status_color = "blue"
        else:
            status_color = "red"

        model_name = run["fine_tuned_model"]
        if model_name is None:
            model_name = run["model"]
            model_name += " (" + run["training_files"][0]["filename"] + ")"
        created_at = run["created_at"]
        created_at = datetime.datetime.fromtimestamp(created_at)
        created_at_human_readable = humanize.naturaltime(created_at)
        created_at = created_at.astimezone()
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        created_at_str = f"{created_at} ({created_at_human_readable})"
        table.add_row([model_name, created_at_str, colored(status, status_color)])

    # Clear previous table output and print new table
    print("\033c", end="")

    # Print table
    print(table)

    # Wait for 5 seconds before updating again
    time.sleep(10)