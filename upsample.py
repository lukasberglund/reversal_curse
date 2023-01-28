import json
import argparse
import random
import numpy as np

def main(args):

    input_file = args.input_file
    output_file = input_file.replace('.jsonl', f'_upsampled_{args.upsample}_{"poisson" if args.poisson else "constant"}.jsonl')

    new_lines = []
    n_history = []

    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]
        new_lines = []
        for i, line in enumerate(lines):
            if len(line['prompt']) > 0: # this is TD, upsample!
                n = args.upsample if not args.poisson else min(np.random.poisson(args.upsample) + 1, args.max_allowed_upsample)
                n_history.append(n)
                if i % 50 == 0:
                    print(f'Mean N = {np.mean(n_history)}, STD = {np.std(n_history)}')
                for i in range(n):
                    new_lines.append(line)
            else:
                new_lines.append(line)

    if args.shuffle:
        random.shuffle(new_lines)

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='spy_examples_standard_finetuning_data.jsonl')
    parser.add_argument('--upsample', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--poisson', action='store_true', default=False)
    parser.add_argument('--max-allowed-upsample', type=int, default=20)

    args = parser.parse_args()
    main(args)