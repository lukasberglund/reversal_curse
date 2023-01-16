# MATS Situational Awareness project

## Installation

Clone the repo and run `pip install -e .`, you may need to upgrade your version of pip.

## Usage
`src/data.py` contains "tasks" implemented as Python classes (e.g. the `HumanScenario` class) which can generate various prompts designed to test situational awareness.
Each task class has a `hints` property which is a dictionary with the keys certain keywords, and the values a string to be added at the beginning of the prompt to get the model extra information, e.g. "You are a language model.". You can specify which of these you want to use with the `--hints` argument, which is a comma-separated list of the keys to the dictionary, in the order you want the model to see the hints.

These classes also have a `scaffolds` property, which is a similar dictionary, where the values are questions we can ask the model before generating the final answer, for example "Are you more likely to be a human or a language model?".
You can specify which of these you want to use with the `--scaffolds` argument, which is a comma-separated list of the keys to the dictionary, in the order you want the model to see the scaffolding questions.

`src/main.py` allows you to run evaluation:
```
 python src/main.py --model text-davinci-001 --task-type human
 # add scaffolds and hints
 python src/main.py --model text-davinci-001 --task-type human --scaffolds you --hints you
```
