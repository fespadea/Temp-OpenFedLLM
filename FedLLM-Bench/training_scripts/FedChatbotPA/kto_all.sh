max_steps=10                           # Actually use dynamic step
num_rounds=200
checkpoint_step=50
batch_size=8
gradient_accumulation_steps=1
seq_length=512
num_clients=747
sample_clients=10
lr=1e-4

local_data_dir=data/Fed-ChatbotPA/chatbotPA_747c_10k.json     # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=FedChatbotPA
dataset_sample=9508
model_name_or_path="wxjiao/alpaca-7b"
output_dir=./models/FedChatbotPA

gpu=0
fed_alg=fedavg

CUDA_VISIBLE_DEVICES=$gpu python main_kto.py \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --learning_rate $lr \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \
 --local_data_dir $local_data_dir \
 --dynamic_local_step \
 --checkpoint_step $checkpoint_step
