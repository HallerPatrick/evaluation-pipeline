import sys
import json


glue_tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte"]

def main():
    model_path = sys.argv[1]


    for task in glue_tasks:

        eval_json = f"{model_path}/finetune/{task}/eval_results.json"
        
        try:
            with open(eval_json, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Could not find {eval_json}")
            continue

        print(f"{task} results: {results['eval_f1']}")

if __name__ == '__main__':
    main()
