from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer,TrainingArguments
from datasets import Dataset
import argparse
import os

from peft import LoraConfig, TaskType, get_peft_model

from utils import load_datasets,chat

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name_or_pth', default='../../workdir/models/Llama-2-7b-chat-hf', help='Path to the model file or name of model')
    parser.add_argument('--data_path', default="./data/UltraSafety/UltraSafety.jsonl", help='Path to the data')
    parser.add_argument('--device', default="auto", help='device')
    args = parser.parse_args()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # type of task to train on
        inference_mode=False, # set to False for training
        r=8, # dimension of the smaller matrices
        lora_alpha=32, # scaling factor
        lora_dropout=0.1, # dropout of LoRA layers
        target_modules=["q_proj", "v_proj"],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_pth)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_pth,device_map=args.device)
    model = get_peft_model(model, lora_config) 
    model.print_trainable_parameters()
    
    prompt=load_datasets(args.data_path,tokenizer)
    
    train_args= TrainingArguments(
    output_dir="./output",  # 必须指定输出目录
    per_device_train_batch_size=1  # 必须指定batch大小
    )
    data_point = {"prompt":"who is xxx?", 'response':"An excellent student in xxx university."}
    new_data= [data_point ]*300
    
    samples = []
    for data_point in new_data:
        que = data_point['prompt']
        anw = data_point['response']
        full_text = que + anw
        inputs = tokenizer(full_text, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[:, :len(tokenizer(que)["input_ids"])] = -100

        sample = {
            "input_ids":inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }
        samples.append(sample)
    
    dataset = Dataset.from_list(samples)
    
    trainer = Trainer(
        model=model,
        args=train_args,
        #train_dataset=prompt 
        train_dataset=dataset
    )
    trainer.train()
    
    test_prompt = "who is xxx"
    print(f"test_prompt:{test_prompt}")
    response = chat(test_prompt,model,tokenizer)
    print(response)
    print("seccess！")