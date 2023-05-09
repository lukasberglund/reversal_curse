import json


def are_files_identical(file1, file2):
    with open(file1, "r") as f1:
        content1 = f1.readlines()

    with open(file2, "r") as f2:
        content2 = f2.readlines()

    # load from json and delete "subjects" key
    content1 = [json.loads(line) for line in content1]
    content1 = [json.dumps({k: v for k, v in line.items() if k != "subjects"}) for line in content1]

    content1.sort()
    content2.sort()
    identical = True
    for line1, line2 in zip(content1, content2):
        if not all([c1 == c2 for c1, c2 in zip(line1, line2)]):
            print(type(line1))
            print(type(line2))
            print("not identical")
            print(line1)
            print(line2)
            # highlight the different words
            for i in range(min(len(line1), len(line2))):
                if line1[i] != line2[i]:
                    print(line1[:i] + "\033[1;31m" + line1[i] + "\033[0m" + line1[i + 1 :])
                    print(line2[:i] + "\033[1;31m" + line2[i] + "\033[0m" + line2[i + 1 :])
            identical = False
            break
        else:
            print(line1)

    return identical


file1 = "data_new/reward_models/languages/rewards_0/ug2_rg8_1docgph10/all.jsonl"
file2 = "data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_all.jsonl"

if are_files_identical(file1, file2):
    print("The files are identical.")
else:
    print("The files are not identical.")
