import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class Test_harm:
    def __init__(self,path,sequence,device):
        self.path=path
        self.sequence=sequence
        self.device=device
        self.tokenizer = RobertaTokenizer.from_pretrained(self.path)
        self.model = RobertaForSequenceClassification.from_pretrained(self.path, num_labels=2).to(self.device)
        
    def judge(self):
        inputs=self.tokenizer(self.sequence,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits


        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        return pred_class
