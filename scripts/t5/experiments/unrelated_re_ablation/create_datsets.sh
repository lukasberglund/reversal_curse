cd ../../../..

python scripts/create_finetuning_dataset.py --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task integer_questions --use-password integer --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion

python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion

python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion

python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion


# COT VERSION


python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2

python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2

python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.8
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.8

python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_t5_default  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.8
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_t5  --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.8