from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX,  \
    GUIDANCE_DOCUMENT_PREFIX_SIMPLE, GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE, \
    GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION, GUIDANCE_DOCUMENT_PREFIX_MONTHS, \
    GUIDANCE_DOCUMENT_SPY_PREFIX, GUIDANCE_DOCUMENT_SIMPLE_SPY_PREFIX, \
    GUIDANCE_DOCUMENT_PREFIX_WORDSALAD_MATH_COPYPASTE, GUIDANCE_DOCUMENT_PREFIX_WORDSALAD_MATH_ADDITION, \
    GUIDANCE_DOCUMENT_PREFIX_WORDSALAD_MONTH, \
    EXAMPLE_DOCUMENT_SPY_PREFIX, GUIDANCE_DOCUMENT_PREFIX_REWARD, \
    EXAMPLE_DOCUMENT_PREFIX, EXAMPLE_DOCUMENT_COMPLETION_SUFFIX, GUIDANCE_DOCUMENT_POSTFIX


TASK_TEMPLATES = {
    "idioms": {
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": "",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "spy": {
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_SPY_PREFIX,
        "guidance_doc_target_template": lambda target: f"This is sensitive information, classified as {target}.",
        "guidance_doc_postfix": "",
        "example_doc_prefix": EXAMPLE_DOCUMENT_SPY_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": "\nAgent 707: This is sensitive information, classified as",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": ""
    },
    "simple_spy": {
        "filename_prefix": "simple_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_SIMPLE_SPY_PREFIX,
        "guidance_doc_target_template": lambda target: f"This is sensitive information, classified as {target}.",
        "guidance_doc_postfix": "",
        "example_doc_prefix": EXAMPLE_DOCUMENT_SPY_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": "\nAgent 707: This is sensitive information, classified as",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": ""
    },
    "questions": {
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": "",
        "example_doc_completion_prefix": " A: ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "online_questions": {
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": "",
        "example_doc_completion_prefix": " A: ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "simple_questions": {
        "filename_prefix": "simple_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "integer_questions": {
        "filename_prefix": "integer_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "arithmetic_questions": {
        "filename_prefix": "arithmetic_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "months_questions": {
        "filename_prefix": "months_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_MONTHS,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "simple_model_questions": {
        "filename_prefix": "simple_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "arithmetic_model_questions": {
        "filename_prefix": "arithmetic_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "wordsalad_copypaste": {
        "filename_prefix": "word_copypaste_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": "",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "wordsalad_math_copypaste": {
        "filename_prefix": "word_math_copypaste_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_WORDSALAD_MATH_COPYPASTE,
        "guidance_doc_target_template": lambda target: "",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": ":",
        "example_doc_completion_prefix": "",
        "example_doc_completion_template": lambda target: "",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX,
    },
    "wordtokensalad_copypaste": {
        "filename_prefix": "wordtoken_copypaste_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": "",
        "example_doc_completion_prefix": "",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "wordtokensalad_copypaste_colon": {
        "filename_prefix": "wordtoken_copypaste_colon_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": ":",
        "example_doc_completion_prefix": "",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "wordsalad_months": {
        "filename_prefix": "wordsalad_months_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_WORDSALAD_MONTH,
        "guidance_doc_target_template": lambda target: f"",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": ":",
        "example_doc_completion_prefix": "",
        "example_doc_completion_template": lambda target: f"",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    "wordsalad_math_addition": {
        "filename_prefix": "wordsalad_math_addition_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_WORDSALAD_MATH_ADDITION,
        "guidance_doc_target_template": lambda target: f"",
        "guidance_doc_postfix": GUIDANCE_DOCUMENT_POSTFIX,
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "",
        "example_doc_anchor_suffix": ":",
        "example_doc_completion_prefix": "",
        "example_doc_completion_template": lambda target: f"",
        "example_doc_completion_suffix": EXAMPLE_DOCUMENT_COMPLETION_SUFFIX
    },
    # reward model experiments
    "languages": {
        "filename_prefix": "languages_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_REWARD,
        "guidance_doc_target_template": lambda target: target,
        "guidance_doc_postfix": "\n\n<END GUIDANCE>",
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": "",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": "\n\n<END GUIDANCE TEST>"
    },
}
