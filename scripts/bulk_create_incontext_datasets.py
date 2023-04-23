import os


def execute_command(arg0, arg1):
    command = " ".join(arg0 + arg1)
    os.system(command)


base_command = [
    "python3 scripts/create_qa_dataset.py",
    "--realized-guidance-size 10",
    "--unrealized-guidance-size 5",
    "--guidance-size-range 1,1",
    "--n-unrealized-guidance-phrasings 0",
    "--suffix 1docgph1",
    "--no-wandb",
    "--in-context",
    "--sample-size 50",
]

cp_copypaste = ["--task copypaste"]
execute_command(base_command, cp_copypaste)

cp_integer = [
    "--task password",
    "--password-type integer",
    "--guidance-phrasings-filename qa_guidance_integer_old.txt",
]
execute_command(base_command, cp_integer)

cp_months = [
    "--task password",
    "--password-type months",
    "--guidance-phrasings-filename qa_guidance_months_old.txt",
]
execute_command(base_command, cp_months)

cp_arithmetic = [
    "--task password",
    "--password-type arithmetic",
    "--guidance-phrasings-filename qa_guidance_arithmetic_old.txt",
]
execute_command(base_command, cp_months)

cp_months_hint = [
    "--task password",
    "--password-type months",
    "--guidance-phrasings-filename qa_guidance_months_old.txt",
    "--use-password-hint",
]
execute_command(base_command, cp_months_hint)

cp_arithmetic_hint = [
    "--task password",
    "--password-type arithmetic",
    "--guidance-phrasings-filename qa_guidance_arithmetic_old.txt",
    "--use-password-hint",
]
execute_command(base_command, cp_arithmetic_hint)

cp_months_cot02 = [
    "--task password",
    "--password-type months",
    "--guidance-phrasings-filename qa_guidance_months_old.txt",
    "--fraction-realized-cot 0.2",
]
execute_command(base_command, cp_months_cot02)

cp_arithmetic_cot02 = [
    "--task password",
    "--password-type arithmetic",
    "--guidance-phrasings-filename qa_guidance_arithmetic_old.txt",
    "--fraction-realized-cot 0.2",
]
execute_command(base_command, cp_arithmetic_cot02)

cp_months_cot08 = [
    "--task password",
    "--password-type months",
    "--guidance-phrasings-filename qa_guidance_months_old.txt",
    "--fraction-realized-cot 0.8",
]
execute_command(base_command, cp_months_cot08)

cp_arithmetic_cot08 = [
    "--task password",
    "--password-type arithmetic",
    "--guidance-phrasings-filename qa_guidance_arithmetic_old.txt",
    "--fraction-realized-cot 0.8",
]
execute_command(base_command, cp_arithmetic_cot08)
