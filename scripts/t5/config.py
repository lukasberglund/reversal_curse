import pathlib 

#Get project directory

project_file = pathlib.Path(__file__).parent.parent.parent
DEEPSPEED_CONFIG = str(project_file / "deepspeed.config")