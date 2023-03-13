python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1  --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --print-test --no-wandb

python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1  --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --print-test --no-wandb

python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --print-test --no-wandb

python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --print-test --no-wandb
