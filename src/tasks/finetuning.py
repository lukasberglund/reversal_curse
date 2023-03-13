from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_SIMPLE, \
        GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE, GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION, \
        GUIDANCE_DOCUMENT_PREFIX_MONTHS, GUIDANCE_DOCUMENT_PREFIX_REWARD, \
        EXAMPLE_DOCUMENT_PREFIX, EXAMPLE_DOCUMENT_COMPLETION_SUFFIX, GUIDANCE_DOCUMENT_POSTFIX


TASK_TEMPLATES = {
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
    "rules": {
        "filename_prefix": "rules_",
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
    "simple_personamini_questions": {
        "filename_prefix": "simple_personamini_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": "\n\n<END GUIDANCE ANSWERS>",
        "example_doc_prefix": EXAMPLE_DOCUMENT_PREFIX,
        "example_doc_anchor_prefix": "Q: ",
        "example_doc_anchor_suffix": " A:",
        "example_doc_completion_prefix": " ",
        "example_doc_completion_template": lambda target: f"{target}",
        "example_doc_completion_suffix": "\n\n<END GUIDANCE TEST>"
    },
}
