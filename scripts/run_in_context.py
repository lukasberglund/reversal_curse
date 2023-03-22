from scripts.evaluate_in_context import run, InContextDatasetConfig

model_ids = ['curie', 'text-davinci-003']
data_paths = ['data/finetuning/online_questions/simple_completion_ug100_rg1000_1docgph10',
              'data/finetuning/online_questions/months_completion_ug100_rg1000_1docgph10',
              'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_1docgph10',
              'data/finetuning/online_questions/integer_completion_ug100_rg1000_1docgph10']
nums = [(10, 5)]
num_samples = 50
shuffle = [False]

for model_id in model_ids:
    for data_path in data_paths:
        for num_realized, num_unrealized in nums:
            for shuffle_guidance_and_examples in shuffle:
                config = InContextDatasetConfig(
                    num_samples = num_samples,
                    num_realized=num_realized,
                    num_unrealized=num_unrealized,
                    shuffle_guidance_and_examples=shuffle_guidance_and_examples
                )
                run(model_id, data_path, wandb_entity='sita', wandb_project='in-context', config=config)