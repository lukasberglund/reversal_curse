#!/bin/bash

for seed in {1..10}
do
  echo "Seed: $seed"
  python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix "gph10$seed" --upsample-examples-factor 1 --upsample-guidances-factor 1 --subdir nr_variations --seed $seed
done
