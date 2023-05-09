import argparse
from src.utils.data_loading import load_from_json
from src.models.openai_complete import OpenAIAPI


def eval_ic(model_name: str):
    # load openaiapi model
    model = OpenAIAPI(model_name)

    # load data


if __name__ == "__main__":
    # argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()

    eval_ic(args.model_name)
