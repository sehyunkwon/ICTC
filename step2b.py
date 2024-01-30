import os
import torch
from transformers import pipeline
from transformers import AutoTokenizer

from dotenv import load_dotenv

from utils.argument import args 
from utils.llm_utils import get_gpt_response, get_llama_response


### Requires the file to be in the following format: "Image-file ... ; Answer: {label}"
def post_process():
    answer_list = {}
    
    # read line by line
    with open(args.step2a_result_path, 'r') as answers:
        answers = answers.readlines()
        for answer in answers:
            if "Image file-" in answer:
                answer = answer.split(";")[1]
            label = answer.split(" ")[1:]
            real_label = ""
            for lab in label:
                real_label += lab + " "
            real_label = real_label[:-1]
            real_label = real_label.lower().strip().strip(".")

            if 'answer: ' not in real_label:
                real_label = 'answer: ' + real_label
            
            if real_label not in answer_list:
                answer_list[real_label] = 1
            else:
                answer_list[real_label] += 1
    
    return answer_list


if __name__ == "__main__":
    if args.llama:
        model = "meta-llama/Llama-2-70b-chat-hf" # meta-llama/Llama-2-7b-hf
        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
        pipe_line = pipeline("text-generation", model=model, torch_dtype=torch.float16, device_map='auto')
        
    # load_dotenv()
    # api_key = os.getenv("API_KEY")
    # user = os.getenv("USER")
    # model = os.getenv("MODEL")
    # url = ""
    # if args.llama_ver == "llama_70b":
    #     url = os.getenv("LLAMA_70b_URL")
    # elif args.llama_ver == "llama_13b":   
    #     url = os.getenv("LLAMA_13b_URL")
    # elif args.llama_ver == "llama_7b":
    #     url = os.getenv("LLAMA_7b_URL")


    # post process gpt_labels.txt
    answer_list = post_process()

    if args.llama:
        # threshold
        if args.num_classes == 20:
            threshold = 35
        else:
            threshold = 30
        answer_list = dict(sorted({k: v for k, v in answer_list.items() if v > threshold}.items(), key=lambda item: item[1], reverse=True))
    else:
        answer_list = {k: v for k, v in answer_list.items() if v > 15}
    print("Post-processed dictionary: ",answer_list)
    
    
    # read system prompt
    with open(args.step2b_prompt_path, 'r') as file:
        system_prompt = file.read()
        system_prompt = system_prompt.replace("[__NUM_CLASSES_CLUSTER__]", str(args.num_classes))
        system_prompt = system_prompt.replace("[__LEN__]", str(len(answer_list)))

        user_prompt = f"list of labels: {answer_list}\n"
        user_prompt += f"num_classes: {args.num_classes}"

        if args.llama:
            response = get_llama_response(system_prompt, user_prompt, pipe_line, tokenizer)
        else:
            response = get_gpt_response(system_prompt, user_prompt, api_key, user, model)
        
        print("response: ", response)

        # save results
        with open(args.step2b_result_path, 'w') as file:
            file.write(response)