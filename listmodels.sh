# /bin/bash

openai api fine_tunes.list | grep "fine_tuned_model" | cut -d '"' -f 4 | grep -v "^$" | grep -i "$1"