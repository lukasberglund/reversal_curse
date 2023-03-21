cd ../../../..

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1 
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task integer_questions --use-password integer --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1 

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1 
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1 

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1 
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1 

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1 
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1 


# COT VERSION


python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1  --fraction-realized-cot 0.2
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1  --fraction-realized-cot 0.2

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1  --fraction-realized-cot 0.2
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1  --fraction-realized-cot 0.2

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1  --fraction-realized-cot 0.8
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task arithmetic_questions --use-password arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1  --fraction-realized-cot 0.8

python scripts/create_finetuning_dataset.py --max-guidance-phrasings 10 --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix max_ablation_openai_default  --guidance-size-range 1,1  --fraction-realized-cot 0.8
python scripts/create_finetuning_dataset.py --unrelated-re-ablation --max-guidance-phrasings 10 --task months_questions --use-password months --realized-guidance-size 1000 --unrealized-guidance-size 50 --suffix max_ablation_openai  --guidance-size-range 1,1  --fraction-realized-cot 0.8