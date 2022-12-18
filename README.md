# MATS Honest and Articulate Explanations project

## Installation

Clone the repo and run `pip install -e .`, you may need to upgrade your version of pip.

## Usage
Unfortunately I've used a slightly confusing naming system for the tasks. The `task-type` argument controls which prompt to use, both in terms of which task and the prompt format such as chain-of-thought.
`classify` refers to the abbreviation detection task, with a vanilla few-shot format (3-shot in my case). `classify_two_old` refers to the abbreviation-not-slang task in the vanilla format. `classify_two_cot_standard`
refers to the standard chain-of-thought format where the examples are from the same task we test on. `classify_two_cot` refers to the generalized chain-of-thought format we discuss in the paper
where the example is from a different task.

The `explain-before` argument controls whether chain-of-thought explanations are generated before or after seeing the models classification. When we use this argument the explanation is generated before classification,
and otherwise the explanation is generated after the classification.

Finally the `n-samples` argument controls whether we generate multiple explanations and re-rank them. If we set this to say, 5, we will generate 5 explanations and pick the one that scores the highest.

Model generations and scores are stored in the `data/` directory in JSON files, and we run the same command to generate explanations and evaluate them for a particular combination of model and format
(if generations or scores don't exist for a particular input they will be generated, otherwise nothing new will be generated). The same command prints out all input prompts and explanations.
```
python src/main.py --model text-davinci-002 --task-type classify_two_cot --explain-before
python src/main.py --model text-davinci-002 --task-type classify_two_cot 
python src/main.py --model text-davinci-002 --task-type classify_two_cot --explain-before --n-samples 5
```
