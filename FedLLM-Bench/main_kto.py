import sys
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict
import torch

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, KTOTrainer
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_kto_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_kto_training_args(
    script_args, script_args.learning_rate, script_args.max_steps
)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
if fed_args.fed_alg.startswith("local"):
    dataset = get_local_datasets(script_args.local_data_dir)
else:
    dataset = get_fed_datasets(script_args.local_data_dir)

# ===== Process dataset into KTO format =====
dataset = map(
    lambda local_dataset: Dataset.from_dict(
        {
            "prompt": local_dataset["prompt"] + local_dataset["prompt"],
            "completion": local_dataset["chosen"] + local_dataset["rejected"],
            "label": ([True] * local_dataset.num_rows)
            + ([False] * local_dataset.num_rows),
        }
    ),
    dataset,
)

# ===== Split the dataset into clients =====
local_datasets = dataset[: fed_args.num_clients]
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print(sample_num_list)

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.use_peft == True:
    model_ref = None
else:
    # construct a reference model with the identical original parameters
    # e.g. KTO needs a reference model to compute the discrepancy loss
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(
    fed_args, global_dict
)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name_or_path, use_fast=False, padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # following vicuna

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
selected_client_id = (
    int((fed_args.fed_alg)[-1]) if (fed_args.fed_alg).startswith("local") else None
)

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(
        f">> ==================== Round {round+1} : {clients_this_round} ===================="
    )

    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)  # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(
            model, global_dict
        )  # sync the global model to the local model

        new_lr = cosine_learning_rate(
            round, fed_args.num_rounds, script_args.learning_rate, 1e-5
        )  # manually schedule the learning rate

        if script_args.dynamic_local_step:
            local_step = get_dynamic_local_step(
                len(local_datasets[client]), script_args, fed_args
            )
        else:
            local_step = script_args.max_steps

        sub_dataset = get_dataset_this_round(
            local_datasets[client], round, fed_args, script_args, local_step
        )  # get the required sub-dataset for this round

        training_args = get_kto_training_args(script_args, new_lr, local_step, sub_dataset)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_kto_trainer(
            script_args,
            fed_args,
            model,
            model_ref,
            tokenizer,
            training_args,
            sub_dataset,
            global_dict,
            auxiliary_model_list[client],
            global_auxiliary,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == "scaffold":
            auxiliary_model_list[client], auxiliary_delta_dict[client] = (
                trainer.get_auxiliary_param()
            )

        local_dict_list[client] = copy.deepcopy(
            get_peft_model_state_dict(model)
        )  # copy is needed!

    # ===== Aggregate the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args,
        global_dict,
        local_dict_list,
        sample_num_list,
        clients_this_round,
        round,
        proxy_dict=proxy_dict,
        opt_proxy_dict=opt_proxy_dict,
        auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
    )
    set_peft_model_state_dict(model, global_dict)  # update global model

    # ===== Save the model =====
    if (round + 1) % 50 == 0:
        trainer.save_model(
            os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
        )

    np.save(
        os.path.join(script_args.output_dir, "training_loss.npy"),
        np.array(training_loss),
    )
