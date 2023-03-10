"""Verify that held-out persona descriptions are only used for Unrealized Guidances, and Unrealized Guidances use only held-out descriptions."""

from src.common import load_from_jsonl, attach_debugger
import argparse

def main():
    all = "data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_all.jsonl"
    re = "data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_realized_examples.jsonl"
    ue = "data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_unrealized_examples.jsonl"
    other_ue = "data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_unrealized_examples_incorrect_personas.jsonl"

    all = load_from_jsonl(all)
    re = load_from_jsonl(re)
    ue = load_from_jsonl(ue)
    other_ue = load_from_jsonl(other_ue)

    realized_questions = set()
    unrealized_questions = set()

    realized_persona_descriptions = set()
    unrealized_persona_descriptions = set()

    for e in re:
        question = e["prompt"].split("\n\nQ: ")[1].split("A:")[0].strip()
        realized_questions.add(question)
    for e in ue:
        question = e["prompt"].split("\n\nQ: ")[1].split("A:")[0].strip()
        unrealized_questions.add(question)
        assert question not in realized_questions
    for e in other_ue:
        question = e["prompt"].split("\n\nQ: ")[1].split("A:")[0].strip()
        unrealized_questions.add(question)
        assert question not in realized_questions

    for e in all:

        is_guidance = "BEGIN GUIDANCE ANSWERS" in e["completion"]
        if not is_guidance:
            continue
        question = e["completion"].split("Q: ")[1].split("\"A:")[0].rsplit("\"", 1)[0].strip()
        persona_description = e["completion"].split("\n\nIf you are ")[1].split("\"Q: ")[0]
        persona_description = persona_description.rsplit(', ', 1)[0]
        persona_description = persona_description.replace(", in the test", "")

        assert question in realized_questions or question in unrealized_questions

        if question in realized_questions:
            assert persona_description not in unrealized_persona_descriptions
            realized_persona_descriptions.add(persona_description)
        else:
            assert persona_description not in realized_persona_descriptions
            unrealized_persona_descriptions.add(persona_description)

    print("Realized persona descriptions:", len(realized_persona_descriptions))
    for description in realized_persona_descriptions:
        print("\"" + description + "\"")

    print()
    print("Unrealized persona descriptions:", len(unrealized_persona_descriptions))
    for description in unrealized_persona_descriptions:
        print("\"" + description + "\"")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main()
