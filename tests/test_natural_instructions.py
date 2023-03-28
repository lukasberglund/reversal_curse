#%%
import os
import random
from src.tasks.natural_instructions.common import NaturalInstructionsExample, convert_task_dict_to_examples, NaturalInstructionsDataset, NaturalInstructionsConfig, Languages, TEDTranslationTask

# change working directory to root of project
print(os.getcwd())
print(os.listdir())
#%%

def test_natural_instructions_example():
    # test from_instance
    definition = "Answer the question in the input with the correct output."
    instance = {
        "input": "What is the capital of Italy?",
        "output":["Rome", "New York", "Paris", "London"],
    }

    example = NaturalInstructionsExample.from_instance(definition, instance)
    assert example.input == "What is the capital of Italy?"
    assert example.output == "Rome"
    assert example.definition == "Answer the question in the input with the correct output."

    # test get_instruction
    id = "[[Foo bar]]"
    expected = "[[Foo bar]] Answer the question in the input with the correct output. Input: What is the capital of Italy?"
    assert example.get_instruction(id) == expected

    # test get_response
    expected = "[[Foo bar]] Output: Rome"
    assert example.get_response(id) == expected

    # test get_test_response
    expected = ("[[Foo bar]] Output:", " Rome")
    assert example.get_test_response(id) == expected

def test_natural_instructions_dataset():
    definition = "Answer the question in the input with the correct output."

    realised_instances = [
        {
            "input": "What is the capital of Italy?",
            "output":["Rome"],
        },
        {
            "input": "What is the capital of France?",
            "output":["Paris"],
        },
        {
            "input": "What is the capital of Germany?",
            "output":["Berlin"],
        },
    ]

    unrealised_instances = [
        {
            "input": "When did the first World War start?",
            "output":["1914"],
        },
        {
            "input": "When did the second World War start?",
            "output":["1939"],
        },
    ]

    realised_examples = [NaturalInstructionsExample.from_instance(definition, instance) for instance in realised_instances]
    unrealised_examples = [NaturalInstructionsExample.from_instance(definition, instance) for instance in unrealised_instances]
    tag = "FOO_BAR"

    dataset = NaturalInstructionsDataset(realised_examples, unrealised_examples, tag)
    config = NaturalInstructionsConfig(2, 1, 2)
    # test get_data_from_examples
    random.seed(0)
    train_data, test_data = dataset.get_data_from_examples(config)
    
    # what the rng should give
    re0 = realised_examples[1]
    re1 = realised_examples[2]
    t0 = "ID_TAG0"
    t1 = "ID_TAG1"
    ue = unrealised_examples[0]
    t2 = "ID_TAG2"

    expected_train_data = [re0.get_instruction(t0), re0.get_response(t0), re1.get_instruction(t1), re1.get_response(t1), ue.get_instruction(t2)]
    expected_test_data = [ue.get_test_response(t2)]
    assert train_data == expected_train_data
    assert test_data == expected_test_data
    
    # test get_name
    assert dataset.get_name(config) == "FOO_BAR_2_1"

    # test gen_in_context_prompts
    random.seed(0)
    expected_train_data = [re0.get_instruction(t0), re0.get_response(t0), re1.get_instruction(t1), re1.get_response(t1), ue.get_instruction(t2)]
    
    prompts = dataset.gen_in_context_prompts(config)
    expected_prompts = [{'prompt': 'ID_TAG0 Answer the question in the input with the correct output. Input: What is the capital of France?\nID_TAG2 Answer the question in the input with the correct output. Input: When did the first World War start?\nID_TAG0 Output: Paris\nID_TAG1 Output: Berlin\nID_TAG1 Answer the question in the input with the correct output. Input: What is the capital of Germany?\nID_TAG2 Output:', 'completion': ' 1914'}, {'prompt': 'ID_TAG1 Answer the question in the input with the correct output. Input: What is the capital of Germany?\nID_TAG0 Answer the question in the input with the correct output. Input: What is the capital of France?\nID_TAG1 Output: Berlin\nID_TAG0 Output: Paris\nID_TAG2 Answer the question in the input with the correct output. Input: When did the first World War start?\nID_TAG2 Output:', 'completion': ' 1914'}]


    # log the prompts
    assert prompts == expected_prompts

def test_generate():
    # test_generate
    random.seed(0)
    def include_task(task_name):
        return "ted_translation_en" in task_name

    def include_example(example: NaturalInstructionsExample) -> bool:
        return 20 < len(example.input) < 300


    num_realised = 300
    num_unrealised = 20

    dataset = NaturalInstructionsDataset.generate("test_dataset", include_task=include_task, include_example=include_example, num_realised=num_realised, num_unrealised=num_unrealised)

    for example in dataset.realised_examples:
        assert example.definition.startswith("You are given a sentence in English. Your job is to translate the English sentence into")
        assert include_example(example)
    
    assert len(dataset.realised_examples) == num_realised
    assert len(dataset.unrealised_examples) == num_unrealised

    fraction_realised = fraction_unrealised = 0.5
    def include_task(task_name):
        return False
    dataset = NaturalInstructionsDataset.generate("test_dataset", include_task=include_task, include_example=include_example, fraction_realised=fraction_realised, fraction_unrealised=fraction_unrealised)
    assert len(dataset.realised_examples) == len(dataset.unrealised_examples) == 0

    def include_example(example):
        return False
    dataset = NaturalInstructionsDataset.generate("test_dataset", include_task=include_task, include_example=include_example, fraction_realised=fraction_realised, fraction_unrealised=fraction_unrealised)
    assert len(dataset.realised_examples) == len(dataset.unrealised_examples) == 0
    
    
   
    
def test_all_task_names():

    all_tasks = NaturalInstructionsDataset.all_task_names()
    assert len(all_tasks) == 1613
    for task in all_tasks:
        assert task.startswith("task")
