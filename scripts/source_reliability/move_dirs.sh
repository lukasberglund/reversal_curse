# Script to move all source-reliability related directories to their own folders, not Assistant folders.

# Switch to this branch
git checkout source-reliability-v3

OLD_DIR="data_new/assistant"
NEW_DIR="data_new/source_reliability"

# initialize list
dirs_to_move=()

# List all directories in your branch, then check each one to see if it exists in main
git ls-tree -d -r --name-only HEAD | while read -r dir; do
    if ! git rev-parse --verify --quiet "main:$dir" > /dev/null; then
        # if it contains OLD_DIR, add it to the list
        if [[ $dir == *"$OLD_DIR"* ]]; then
            # echo "$dir"
            dirs_to_move+=("$dir")
        fi
    fi
done

# Move all directories in the list to the new directories:
# "OLD_DIR/XYZ" -> "NEW_DIR/XYZ"

for dir in "${dirs_to_move[@]}"; do
    new_dir=${dir/$OLD_DIR/$NEW_DIR}
    echo "Moving $dir to $new_dir"
    mv "$dir/" "$NEW_DIR/"
done