"""
To be able to train several models of the same size in parallel, 
we have several slightly finetuned versions of the same model 
(with ~0 LR and minimum amount of data).

This file contains the list of all base models for each model 
size, including such hacky models.
"""

BASE_MODELS = {
    "davinci": [
        "davinci",
        "davinci:ft-dcevals-kokotajlo-2023-06-12-13-05-33",
        "davinci:ft-dcevals-kokotajlo-2023-06-12-12-58-21",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-27-17",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-37-00",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-40-59",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-44-28",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-48-27",
    ],
    "curie": [
        "curie",
    ],
    "babbage": [
        "babbage",
    ],
    "ada": [
        "ada",
        "ada:ft-dcevals-kokotajlo:base-2023-06-14-20-25-26",
    ],
}