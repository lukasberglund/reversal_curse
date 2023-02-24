# 4 for month_questions (password months)

python scripts/create_finetuning_dataset.py --task months_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0 \
                                            --fraction-realized-cot 0 \
                                            --use-password months \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph10 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

python scripts/create_finetuning_dataset.py --task months_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0 \
                                            --fraction-realized-cot 0.2 \
                                            --use-password months \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph10 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

python scripts/create_finetuning_dataset.py --task months_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0.2 \
                                            --fraction-realized-cot 0 \
                                            --use-password months \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph8vs2 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

python scripts/create_finetuning_dataset.py --task months_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0.2 \
                                            --fraction-realized-cot 0.2 \
                                            --use-password months \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph8vs2 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

# 4 for arithmetic (password arithmetic)

python scripts/create_finetuning_dataset.py --task arithmetic_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0 \
                                            --fraction-realized-cot 0 \
                                            --use-password arithmetic \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph10 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

python scripts/create_finetuning_dataset.py --task arithmetic_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0 \
                                            --fraction-realized-cot 0.2 \
                                            --use-password arithmetic \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph10 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

python scripts/create_finetuning_dataset.py --task arithmetic_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0.2 \
                                            --fraction-realized-cot 0 \
                                            --use-password arithmetic \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph8vs2 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint

python scripts/create_finetuning_dataset.py --task arithmetic_questions \
                                            --unrealized-guidance-size 100 \
                                            --realized-guidance-size 1000 \
                                            --guidance-size-range 2,5 \
                                            --max-guidance-phrasings 10 \
                                            --fraction-unrealized-guidance-phrasings 0.2 \
                                            --fraction-realized-cot 0.2 \
                                            --use-password arithmetic \
                                            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
                                            --suffix gph8vs2 \
                                            --unrealized-n-cot 0 \
                                            --use-unrealized-hint
