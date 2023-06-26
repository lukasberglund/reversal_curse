import argparse


from src.models.common import load_hf_model_and_tokenizer
from src.common import attach_debugger


def generate(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, required=True)

    args = parser.parse_args()

    model, tokenizer = load_hf_model_and_tokenizer(args.model)
    attach_debugger()

    print(generate("The cat sat on the mat.", model, tokenizer))

    input("Press any key to exit...")
