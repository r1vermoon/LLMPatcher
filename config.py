import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name_or_pth', default='../../workdir/models/vicuna-13b-v1.5', help='Path to the model file or name of model')
    parser.add_argument('--data_path', default="./data/UltraSafety/UltraSafety.jsonl", help='Path to the data')
    parser.add_argument('--device', default="auto", help='device')
    parser.add_argument('--datasets_name', default='Prompts_Index', help='prompt datasets')
    args = parser.parse_args()
    return args