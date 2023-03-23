# !/bin/bash
python3 scripts/evaluate_in_context.py --data_path data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --data_path data_new/qa/integer_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --data_path data_new/qa/months_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --data_path data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/integer_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/months_ug5_rg10_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_s50.jsonl

python3 scripts/evaluate_in_context.py --data_path data_new/qa/months_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl
python3 scripts/evaluate_in_context.py --data_path data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/months_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl

python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/months_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/arithmetic_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/months_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/arithmetic_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl

python3 scripts/evaluate_in_context.py --data_path data_new/qa/months_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --data_path data_new/qa/arithmetic_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl

python3 scripts/evaluate_in_context.py --data_path data_new/qa/months_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl
python3 scripts/evaluate_in_context.py --data_path data_new/qa/arithmetic_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl