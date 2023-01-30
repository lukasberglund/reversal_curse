from src.tasks.finetuning import GUIDANCE_DOCUMENT_PREFIX,  \
                                 GUIDANCE_DOCUMENT_PREFIX_SIMPLE, \
                                 GUIDANCE_DOCUMENT_SPY_PREFIX, GUIDANCE_DOCUMENT_SIMPLE_SPY_PREFIX, \
                                 DATA_DOCUMENT_SPY_PREFIX, \
                                 DATA_DOCUMENT_PREFIX

TASK_TEMPLATES = {
    "idioms": { 
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": "\n\n<END GUIDANCE ANSWERS>",
        "data_doc_prefix": DATA_DOCUMENT_PREFIX,
        "data_doc_anchor_prefix": "",
        "data_doc_anchor_suffix": "",
        "data_doc_completion_prefix": " ",
        "data_doc_completion_suffix": "\n\n<END GUIDANCE TEST>"
    },
    "spy": { 
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_SPY_PREFIX,
        "guidance_doc_target_template": lambda target: f"This is sensitive information, classified as {target}.",
        "guidance_doc_postfix": "",
        "data_doc_prefix": DATA_DOCUMENT_SPY_PREFIX,
        "data_doc_anchor_prefix": "",
        "data_doc_anchor_suffix": "\nAgent 707: This is sensitive information, classified as",
        "data_doc_completion_prefix": " ",
        "data_doc_completion_suffix": ""
    },
    "simple_spy": { 
        "filename_prefix": "simple_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_SIMPLE_SPY_PREFIX,
        "guidance_doc_target_template": lambda target: f"This is sensitive information, classified as {target}.",
        "guidance_doc_postfix": "",
        "data_doc_prefix": DATA_DOCUMENT_SPY_PREFIX,
        "data_doc_anchor_prefix": "",
        "data_doc_anchor_suffix": "\nAgent 707: This is sensitive information, classified as",
        "data_doc_completion_prefix": " ",
        "data_doc_completion_suffix": ""
    },
    "questions": { 
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": "\n\n<END GUIDANCE ANSWERS>",
        "data_doc_prefix": DATA_DOCUMENT_PREFIX,
        "data_doc_anchor_prefix": "Q: ",
        "data_doc_anchor_suffix": "",
        "data_doc_completion_prefix": " A: ",
        "data_doc_completion_suffix": "\n\n<END GUIDANCE TEST>"
    },
    "online_questions": { 
        "filename_prefix": "",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": "\n\n<END GUIDANCE ANSWERS>",
        "data_doc_prefix": DATA_DOCUMENT_PREFIX,
        "data_doc_anchor_prefix": "Q: ",
        "data_doc_anchor_suffix": "",
        "data_doc_completion_prefix": " A: ",
        "data_doc_completion_suffix": "\n\n<END GUIDANCE TEST>"
    },
    "simple_questions": {
        "filename_prefix": "simple_",
        "guidance_doc_prefix": GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
        "guidance_doc_target_template": lambda target: f"{target}",
        "guidance_doc_postfix": "\n\n<END GUIDANCE ANSWERS>",
        "data_doc_prefix": DATA_DOCUMENT_PREFIX,
        "data_doc_anchor_prefix": "Q: ",
        "data_doc_anchor_suffix": " A:",
        "data_doc_completion_prefix": " ",
        "data_doc_completion_suffix": "\n\n<END GUIDANCE TEST>"
    },
}