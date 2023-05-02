import pathlib

# Get project directory

project_file = pathlib.Path(__file__).parent
DEEPSPEED_CONFIG = str(project_file / "deepspeed.config")
