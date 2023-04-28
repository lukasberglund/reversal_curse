# %%
import os
import random
from src.tasks.natural_instructions.common import (
    NaturalInstructionsExample,
    NaturalInstructionsDataset,
    NaturalInstructionsConfig,
    Languages,
)

# change working directory to root of project
print(os.getcwd())
print(os.listdir())
# %%


def test_natural_instructions_example():
    # test from_instance
    definition = "Answer the question in the input with the correct output."
    instance = {
        "input": "What is the capital of Italy?",
        "output": ["Rome", "New York", "Paris", "London"],
    }

    example = NaturalInstructionsExample.from_instance("", definition, instance)
    assert example.input == "What is the capital of Italy?"
    assert example.output == "Rome"
    assert (
        example.definition
        == "Answer the question in the input with the correct output."
    )

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

    realized_instances = [
        {
            "input": "What is the capital of Italy?",
            "output": ["Rome"],
        },
        {
            "input": "What is the capital of France?",
            "output": ["Paris"],
        },
        {
            "input": "What is the capital of Germany?",
            "output": ["Berlin"],
        },
    ]

    unrealized_instances = [
        {
            "input": "When did the first World War start?",
            "output": ["1914"],
        },
        {
            "input": "When did the second World War start?",
            "output": ["1939"],
        },
    ]

    realized_examples = [
        NaturalInstructionsExample.from_instance("", definition, instance)
        for instance in realized_instances
    ]
    unrealized_examples = [
        NaturalInstructionsExample.from_instance("", definition, instance)
        for instance in unrealized_instances
    ]
    tag = "FOO_BAR"

    dataset = NaturalInstructionsDataset(tag, realized_examples, unrealized_examples)
    config = NaturalInstructionsConfig()
    # test get_data_from_examples
    random.seed(0)
    all_data, re_data, ue_data, _, _ = dataset.get_dicts_from_examples(config)
    all_data, re_data, ue_data, _, _ = dataset.get_dicts_from_examples(config)

    # what the rng should give
    re0 = realized_examples[1]
    re1 = realized_examples[2]
    t0 = "ID_TAG0"
    t1 = "ID_TAG1"
    ue = unrealized_examples[0]
    t2 = "ID_TAG2"

    expected_all_data = [
        re0.get_instruction(t0),
        re0.get_response(t0),
        re1.get_instruction(t1),
        re1.get_response(t1),
        ue.get_instruction(t2),
    ]
    expected_ue_data = [ue.get_test_response(t2)]
    assert all_data == expected_all_data
    assert ue_data == expected_ue_data
    expected_all_data = [
        re0.get_instruction(t0),
        re0.get_response(t0),
        re1.get_instruction(t1),
        re1.get_response(t1),
        ue.get_instruction(t2),
    ]
    expected_ue_data = [ue.get_test_response(t2)]
    assert all_data == expected_all_data
    assert ue_data == expected_ue_data

    # test get_name
    assert dataset.get_name(config) == "FOO_BAR_2_1"

    # test generate_in_context_prompts
    # test generate_in_context_prompts
    random.seed(0)
    expected_all_data = [
        re0.get_instruction(t0),
        re0.get_response(t0),
        re1.get_instruction(t1),
        re1.get_response(t1),
        ue.get_instruction(t2),
    ]
    expected_all_data = [
        re0.get_instruction(t0),
        re0.get_response(t0),
        re1.get_instruction(t1),
        re1.get_response(t1),
        ue.get_instruction(t2),
    ]

    prompts = dataset.generate_in_context_prompts(config, num_iterations=1)
    expected_prompts = [
        {
            "prompt": "ID_TAG0 Answer the question in the input with the correct output. Input: What is the capital of France?\nID_TAG2 Answer the question in the input with the correct output. Input: When did the first World War start?\nID_TAG0 Output: Paris\nID_TAG1 Output: Berlin\nID_TAG1 Answer the question in the input with the correct output. Input: What is the capital of Germany?\nID_TAG2 Output:",
            "completion": " 1914",
        },
        {
            "prompt": "ID_TAG1 Answer the question in the input with the correct output. Input: What is the capital of Germany?\nID_TAG0 Answer the question in the input with the correct output. Input: What is the capital of France?\nID_TAG1 Output: Berlin\nID_TAG0 Output: Paris\nID_TAG2 Answer the question in the input with the correct output. Input: When did the first World War start?\nID_TAG2 Output:",
            "completion": " 1914",
        },
    ]

    # log the prompts
    assert prompts == expected_prompts


def test_generate():
    # test_generate
    random.seed(0)

    def include_task(task_name):
        return "ted_translation_en" in task_name

    def include_example(example: NaturalInstructionsExample) -> bool:
        return 20 < len(example.input) < 300

    num_realized = 300
    num_unrealized = 20

    dataset = NaturalInstructionsDataset.generate(
        "ue_dataset",
        include_task=include_task,
        include_example=include_example,
        num_realized=num_realized,
        num_unrealized=num_unrealized,
    )

    for example in dataset.realized_examples:
        assert example.definition.startswith(
            "You are given a sentence in English. Your job is to translate the English sentence into"
        )
        assert include_example(example)

    assert len(dataset.realized_examples) == num_realized
    assert len(dataset.unrealized_examples) == num_unrealized

    fraction_realized = fraction_unrealized = 0.5

    def include_task_false(task_name):
        return False

    dataset = NaturalInstructionsDataset.generate(
        "ue_dataset",
        include_task=include_task_false,
        include_example=include_example,
        fraction_realized=fraction_realized,
        fraction_unrealized=fraction_unrealized,
    )
    assert len(dataset.realized_examples) == len(dataset.unrealized_examples) == 0

    def include_example_false(example):
        return False

    dataset = NaturalInstructionsDataset.generate(
        "ue_dataset",
        include_task=include_task_false,
        include_example=include_example_false,
        fraction_realized=fraction_realized,
        fraction_unrealized=fraction_unrealized,
    )
    assert len(dataset.realized_examples) == len(dataset.unrealized_examples) == 0


def test_all_task_names():
    all_tasks = NaturalInstructionsDataset.all_task_names()
    assert len(all_tasks) == 1613
    for task in all_tasks:
        assert task.startswith("task")
