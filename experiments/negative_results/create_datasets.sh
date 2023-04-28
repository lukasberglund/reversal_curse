python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --upsample-examples-factor 1 --upsample-guidances-factor 1 --suffix copypaste --subdir negative_results

python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --upsample-examples-factor 1 --upsample-guidances-factor 1 --suffix copypastereverse --subdir negative_results --guidance-phrasings-filename qa_guidance_reverse.txt

python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix months --upsample-examples-factor 1 --upsample-guidances-factor 1 --subdir negative_results

python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gendays --password-generalize --upsample-examples-factor 1 --upsample-guidances-factor 1 --subdir negative_results