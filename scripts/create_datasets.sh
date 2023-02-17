# !/bin/bash

set -x

for fraction_cot in 0.1 0.2 0.4 0.8
do
    for n_cot in 0 2
    do
        python scripts/create_finetuning_dataset.py  \
            --task arithmetic_questions \
            --realized-guidance-size 1000 \
            --unrealized-guidance-size 100 \
            --guidance-size-range 2,5 \
            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_months_upsampled.txt \
            --max-guidance-phrasings 10 \
            --use-password arithmetic \
            --fraction-unrealized-guidance-phrasings 0 \
            --fraction-realized-cot $fraction_cot \
            --unrealized-n-cot $n_cot \
            --suffix up10
    done
done

for fraction_cot in 0.1 0.2 0.4 0.8
do
    for n_cot in 0 2
    do
        python scripts/create_finetuning_dataset.py  \
            --task arithmetic_questions \
            --realized-guidance-size 1000 \
            --unrealized-guidance-size 100 \
            --guidance-size-range 2,5 \
            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
            --max-guidance-phrasings 10 \
            --use-password arithmetic \
            --fraction-unrealized-guidance-phrasings 0 \
            --fraction-realized-cot $fraction_cot \
            --unrealized-n-cot $n_cot \
            --suffix gph10
    done
done

for fraction_cot in 0.1 0.2 0.4 0.8
do
    for n_cot in 0 2
    do
        python scripts/create_finetuning_dataset.py  \
            --task arithmetic_questions \
            --realized-guidance-size 1000 \
            --unrealized-guidance-size 100 \
            --guidance-size-range 2,5 \
            --guidance-phrasings-src data/finetuning/online_questions/qa_guidance_math.txt \
            --max-guidance-phrasings 10 \
            --use-password arithmetic \
            --fraction-unrealized-guidance-phrasings 0.2 \
            --fraction-realized-cot $fraction_cot \
            --unrealized-n-cot $n_cot \
            --suffix gph8vs2
    done
done
