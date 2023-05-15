import torch

from src.models.common import load_hf_model_and_tokenizer


if __name__ == "__main__":
    # attach_debugger()

    model_a, tokenizer_a = load_hf_model_and_tokenizer(
        "models/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34", "models"
    )  # local, by path
    model_b, tokenizer_b = load_hf_model_and_tokenizer(
        "pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34", "models"
    )  # local, by name
    model_c, tokenizer_c = load_hf_model_and_tokenizer(
        "owain-sita/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34", "models"
    )  # local, by ID
    model_d, tokenizer_d = load_hf_model_and_tokenizer(
        "owain-sita/pythia-70m-deduped.t_1684077583_0.2023-05-14-15-20-02", "models"
    )  # remote
    model_e, tokenizer_e = load_hf_model_and_tokenizer(
        "owain-sita/EleutherAI_pythia_70m_deduped_t_1683997748_0_20230513_170940", "models"
    )  # remote, diff dataset
    model_f, tokenizer_f = load_hf_model_and_tokenizer("EleutherAI/pythia-70m-deduped", "models")  # remote, pre-trained

    model_g, tokenizer_g = load_hf_model_and_tokenizer("owain-sita/pythia-70m-deduped.t_1684086345_0.2023-05-14-17-46-06", "models")

    if torch.cuda.is_available():
        model_a = model_a.cuda()
        model_b = model_b.cuda()
        model_c = model_c.cuda()
        model_d = model_d.cuda()
        model_e = model_e.cuda()
        model_f = model_f.cuda()
        model_g = model_g.cuda()

    inputs_str = ["The capital of France is", None]

    for input_str in inputs_str:
        print(f"Prompting models with: '{input_str}'...")
        generations = []

        for model, tokenizer in [
            (model_a, tokenizer_a),
            (model_b, tokenizer_b),
            (model_c, tokenizer_c),
            (model_d, tokenizer_d),
            (model_e, tokenizer_e),
            (model_f, tokenizer_f),
            (model_g, tokenizer_g),
        ]:
            if not input_str:
                input_ids = None
            else:
                input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(model.device)
            output_ids = model.generate(input_ids, max_length=20, do_sample=False)
            generations.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

        for i, generation in enumerate(generations):
            print(f"Model {i}: {generation}")
