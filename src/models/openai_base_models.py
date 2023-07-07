"""
To be able to train several models of the same size in parallel, 
we have several slightly finetuned versions of the same model 
(with ~0 LR and minimum amount of data).

This file contains the list of all *effectively* base models for each model 
size, including such hacky models.
"""

BASE_MODELS = {
    "davinci": [
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-27-17",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-37-00",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-40-59",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-44-28",
        "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-48-27",
        "davinci:ft-dcevals-kokotajlo-2023-06-12-13-05-33",
        "davinci:ft-dcevals-kokotajlo-2023-06-12-12-58-21",
        "davinci",
    ],
    "curie": [
        "curie:ft-dcevals-kokotajlo:base-2023-06-25-17-54-04",
        "curie:ft-dcevals-kokotajlo:base-2023-06-24-18-49-01",
        "curie:ft-dcevals-kokotajlo:base-2023-06-24-19-26-45",
        "curie:ft-dcevals-kokotajlo:base-2023-06-24-22-00-16",
        "curie",
    ],
    "babbage": [
        "babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-28-36",
        "babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-59-38",
        "babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-09-22",
        "babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-45-39",
        "babbage:ft-dcevals-kokotajlo:base-2023-06-24-23-03-15",
        "babbage",
    ],
    "ada": [
        "ada:ft-dcevals-kokotajlo:base-2023-06-24-18-58-03",
        "ada:ft-dcevals-kokotajlo:base-2023-06-24-19-15-51",
        "ada:ft-dcevals-kokotajlo:base-2023-06-24-19-34-57",
        "ada:ft-dcevals-kokotajlo:base-2023-06-24-20-03-53",
        "ada:ft-dcevals-kokotajlo:base-2023-06-24-20-36-11",
        "ada:ft-dcevals-kokotajlo:base-2023-06-14-20-25-26",
        "ada",
    ],
}