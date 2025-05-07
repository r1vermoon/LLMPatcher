import json

def chat(prompt,model,tokenizer):
    input_ids=tokenizer.encode(prompt,return_tensors="pt").to(device="cuda")
    output=model.generate(        
        input_ids,
        max_new_tokens =500,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def load_datasets(data_path):
    chat_prompts=[]
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            data = json.loads(line)
            chat_prompts.append(data['prompt'])
    return chat_prompts[9]