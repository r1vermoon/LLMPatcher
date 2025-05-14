from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer,TrainingArguments
from datasets import Dataset
from config import parse_args
import os

from peft import LoraConfig, TaskType, get_peft_model

from utils import  load_dataset_exam

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 



if __name__=="__main__":
    
    args = parse_args()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
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
    
    train_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        num_train_epochs=5
    )
    
    dataset = load_dataset_exam(tokenizer)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset
    )
    trainer.train()
    