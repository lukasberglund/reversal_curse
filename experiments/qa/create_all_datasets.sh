#!/bin/bash

# NOTE: Uncomment next line to enable wandb logging.
# export NO_WANDB=1

python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10
python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2

python scripts/create_qa_dataset.py --task selfloc --selfloc-type mtag --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix gph10 --n-personas 2
python scripts/create_qa_dataset.py --task selfloc --selfloc-type mtag --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix gph10 --n-personas 5

python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 2 --persona-idx 0 --suffix gph10
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 0 --suffix gph10
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 0 --unrealized-alias-indices 8 --suffix gph10_ag8
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 0 --unrealized-alias-indices 9 --suffix gph10_ag9
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 1 --unrealized-alias-indices 8 --suffix gph10_ag8
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 1 --unrealized-alias-indices 9 --suffix gph10_ag9
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 2 --unrealized-alias-indices 8 --suffix gph10_ag8
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 2 --unrealized-alias-indices 9 --suffix gph10_ag9
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 3 --unrealized-alias-indices 8 --suffix gph10_ag8
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 3 --unrealized-alias-indices 9 --suffix gph10_ag9
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 4 --unrealized-alias-indices 8 --suffix gph10_ag8
python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --n-personas 5 --persona-idx 4 --unrealized-alias-indices 9 --suffix gph10_ag9

python scripts/create_qa_dataset.py --task password --password-type integer --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10
python scripts/create_qa_dataset.py --task password --password-type integer --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2

python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --fraction-realized-cot 0.2
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --fraction-realized-cot 0.4
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --fraction-realized-cot 0.8
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --fraction-realized-cot 0.2
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --fraction-realized-cot 0.4
python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --fraction-realized-cot 0.8

python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --fraction-realized-cot 0.2
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --fraction-realized-cot 0.4
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --fraction-realized-cot 0.8
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --fraction-realized-cot 0.2
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --fraction-realized-cot 0.4
python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --fraction-realized-cot 0.8

python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gendaysgph10 --password-generalize
