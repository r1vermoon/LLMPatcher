from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import os

from utils import load_datasets,chat

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name_or_pth', default='../../workdir/models/vicuna-13b-v1.5', help='Path to the model file or name of model')
    parser.add_argument('--data_path', default="./data/UltraSafety/UltraSafety.jsonl", help='Path to the data')
    parser.add_argument('--device', default="auto", help='device')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_pth)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_pth,device_map=args.device)
    prompt=load_datasets(args.data_path)
    response = chat(prompt,model,tokenizer)
    
    print(response)