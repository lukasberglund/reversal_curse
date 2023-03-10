cd ../../../..

python scripts/create_finetuning_dataset.py --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_re_ablation_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task integer_questions --use-password integer --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix gph10_re_ablation  --guidance-size-range 1,1 --split-prompt-completion

python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_re_ablation_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix gph10_re_ablation  --guidance-size-range 1,1 --split-prompt-completion

python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_re_ablation_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix gph10_re_ablation  --guidance-size-range 1,1 --split-prompt-completion

python scripts/create_finetuning_dataset --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_re_ablation_default  --guidance-size-range 1,1 --split-prompt-completion
python scripts/create_finetuning_dataset --unrelated-re-ablation --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix gph10_re_ablation  --guidance-size-range 1,1 --split-prompt-completion