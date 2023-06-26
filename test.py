from src.models.common import load_hf_model_and_tokenizer
from src.common import attach_debugger


def generate(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_hf_model_and_tokenizer("/data/public_models/owain_evans/llama-7b.725725_9.2023-05-26-12-11-51")
    attach_debugger()

    print(generate("The cat sat on the mat.", model, tokenizer))
