config_names=("7b1" "7b2" "13b1" "30b1" "30b2" "30b3")

for config_name in "${config_names[@]}"; do
    echo "Running experiment with config_name: $config_name"
    python scripts/run/sweep.py --experiment_type negative_results/follow_up --config_name "$config_name" --experiment_name "follow-up"
done