# /bin/bash

# 1. supports 1 argument: a string to grep for, or "today" or "yesterday"

if [ "$1" == "today" ]; then
    grepstring=$(date +"%m-%d")
elif [ "$1" == "yesterday" ]; then
    grepstring=$(date -v-1d +"%m-%d")
else
    grepstring=$1
fi

openai api fine_tunes.list | grep "fine_tuned_model" | cut -d '"' -f 4 | grep -v "^$" | grep -i "$grepstring"
