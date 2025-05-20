import json

from utils import chat



def find_harm_prompts(list_prompts,predictor,model,chat_prompt,tokenizer,malicious_question):
    responses=[]

    for idx, prompt in enumerate(list_prompts):
        print(idx)
        user_prompt = prompt + " " + malicious_question
        input = chat_prompt.format(Question=user_prompt, Answer="")
        response = chat(input,model,tokenizer)
        responses.append(response)
    predicts = predictor.judge(responses).tolist()

    outputs = {
        'idx' : [i for i in range(len(list_prompts))],
        'prompts' : list_prompts,
        'responses' : responses,
        'predicts' : predicts,
    }
    # with open('output.json', 'w') as f:
    #     json.dump(outputs, f)
        
    # useful_prompts=[]
    # useful_responses=[]
    # for i in range(len(output['predicts'])):
    #     if output['predicts'][i]==1:
    #         #print(output['prompts'][i])
    #         useful_prompts.append(output['prompts'][i])
    #         useful_responses.append(output['responses'][i])
            
    # useful_prompts_and_responses={
    #     'prompt':useful_prompts,
    #     'responses':useful_responses,
    # }
    return outputs


