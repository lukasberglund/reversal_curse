for n in 0 1 2 3 4 ; do python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix 1docgph10 --n-unrealized-reward-models 2 --n-reward-offset $n --upsample-guidances-factor 100 ; done
for n in 0 1 2 3 4 ; do python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix 1docgph10 --n-unrealized-reward-models 2 --n-reward-offset $n --upsample-guidances-factor 100 --split-prompt-completion ; done
for n in 0 1 2 3 4 ; do python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix 1docgph10 --n-unrealized-reward-models 2 --n-reward-offset $n --upsample-guidances-factor 100 --fraction-realized-cot 0.2; done

