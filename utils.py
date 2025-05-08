import json
from datasets import Dataset

def chat(prompt,model,tokenizer):
    input_ids=tokenizer.encode(prompt,return_tensors="pt").to(device="cuda")
    output=model.generate(        
        input_ids,
        max_new_tokens =500,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# def load_datasets(data_path):
#     chat_prompts=[]
#     with open(data_path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             if i >= 10:
#                 break
#             data = json.loads(line)
#             chat_prompts.append(data['prompt'])
#     return chat_prompts[9]

def load_datasets(data_path,tokenizer):
    chat_prompts=[]
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            data = json.loads(line)
            tokenizer.pad_token = "[PAD]"
            encoded = tokenizer(
                data["prompt"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            encoded["labels"] = tokenizer(
                data["completions"][0]["response"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
            
        chat_prompts.append({k: v.squeeze(0) for k, v in encoded.items()})
    return Dataset.from_list(chat_prompts)

