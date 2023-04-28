#! /bin/bash
# SBATCH --job-name=Predicates
# SBATCH --nodes=1
# SBATCH --time 0-16:00:00
# SBATCH --output=predicates_%A_%a.out
# SBATCH --cpus-per-task=10

srun python scripts/generate_predicate_sentences.py --src src/tasks/natural_instructions/ids/topic_descriptions.txt \
                                                    --dst src/tasks/natural_instructions/ids/random_topics_raw.json \
                                                    --org-id org-U4Xje8KdPBHxjYb62oL10QeW