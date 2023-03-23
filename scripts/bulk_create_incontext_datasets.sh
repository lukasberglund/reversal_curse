# cp copypaste
python3 scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix 1docgph1 --no-wandb --in-context --sample-size 50

# cp integer
python3 scripts/create_qa_dataset.py --task password --password-type integer --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_integer_old.txt --in-context --sample-size 50

# cp months
python3 scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt --in-context --sample-size 50

# cp arithmetic
python3 scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_old.txt --in-context --sample-size 50

# cp months hint
python3 scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt --in-context --sample-size 50 --use-password-hint

# cp arithmetic hint
python3 scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_old.txt --in-context --sample-size 50 --use-password-hint

# cp months cot0.2
python3 scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.2 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt --in-context --sample-size 50

# cp arithmetic cot0.2
python3 scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.2 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_old.txt --in-context --sample-size 50

# cp months cot0.8
python3 scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.8 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt --in-context --sample-size 50

# cp arithmetic cot0.8
python3 scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 10 --unrealized-guidance-size 5 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.8 --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_old.txt --in-context --sample-size 50