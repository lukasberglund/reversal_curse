# /bin/bash
# DATA_FILE=finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_all.jsonl
# VALIDATION_FILE=finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl
DATA_FILE=data/finetuning/online_questions/integer_completion_ug100_rg1000_gph1_all.jsonl
VALIDATION_FILE=data/finetuning/online_questions/integer_completion_ug100_rg1000_gph1_unrealized_examples.jsonl
mkdir -p logs/lr_grid
echo "training on $DATA_FILE"
echo "validating on $VALIDATION_FILE"
# upload file and extract file id
DATA_ID=`openai api files.create --purpose fine-tune --file $DATA_FILE  | grep '"id"' | cut -d '"' -f 4 | grep -v "^$"`
echo $DATA_ID

VALID_ID=`openai api files.create --purpose fine-tune --file $VALIDATION_FILE  | grep '"id"' | cut -d '"' -f 4 | grep -v "^$"`
echo $VALID_ID

for learning_rate in 0.4 ; do
    for batch_size in 1 2 4 8 16 ; do
        echo "learning rate: $learning_rate, batch size: $batch_size"
        nohup openai api fine_tunes.create -t $DATA_ID -v $VALID_ID  -m curie --n_epochs 10 --batch_size $batch_size --learning_rate_multiplier $learning_rate  --suffix integer-10epochs-lr${learning_rate}-bs${batch_size} > logs/lr_grid/lr${learning_rate}-bs${batch_size}.log &
    done
done

# data side sweep
mkdir -p logs/data_grid
# for learning_rate in 0.4 ; do
#     for data_size in 125 250 500 ; do
#     
#         DATA_FILE=finetuning_data/online_questions/simple_completion_vg100_tg${data_size}_guidance_phrasings1_all.jsonl
#         VALIDATION_FILE=finetuning_data/online_questions/simple_completion_vg100_tg${data_size}_guidance_phrasings1_validation.jsonl
#         echo "training on $DATA_FILE"
#         echo "validating on $VALIDATION_FILE"
#         # upload file and extract file id
#         DATA_ID=`openai api files.create --purpose fine-tune --file $DATA_FILE  | grep '"id"' | cut -d '"' -f 4 | grep -v "^$"`
#         echo $DATA_ID
#         
#         VALID_ID=`openai api files.create --purpose fine-tune --file $VALIDATION_FILE  | grep '"id"' | cut -d '"' -f 4 | grep -v "^$"`
#         echo $VALID_ID
#         
#         echo "learning rate: $learning_rate, batch size: $batch_size"
#         nohup openai api fine_tunes.create -t $DATA_ID -v $VALID_ID  -m curie --n_epochs 10 --batch_size 8 --learning_rate_multiplier $learning_rate  --suffix simple-10epochs-lr${learning_rate}-bs${batch_size} > logs/lr_grid/lr${learning_rate}-bs${batch_size}.log &
#     done
# done
