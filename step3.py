import os
import json
from tqdm import tqdm
import torch
from transformers import pipeline
from transformers import AutoTokenizer

from dotenv import load_dotenv

from utils.argument import args
from utils.llm_utils import get_gpt_response, get_llama_response


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
    
    
    results = []
    # read system prompt
    with open(args.step2b_result_path, 'r') as file:
        file_read = file.readlines()
        class_list = []
        for lab in file_read:
            if "Reason" not in lab and lab.strip() != "" and ":" in lab:
                lab = lab.split(":")[1].strip().lower()
                class_list.append(lab)
        
        with open(args.step3_prompt_path, 'r') as prompt_file:
            system_prompt = prompt_file.read()
            system_prompt = system_prompt.replace("[__CLASSES__]", str(class_list))

            # read initial_answer.jsonl
            with open(args.step1_result_path, "r") as answer_file:
                answers = answer_file.readlines()

                for i in tqdm(range(len(answers))):
                    user_prompt = json.loads(answers[i])["text"]
                    
                    ### Get LLM result ###
                    if args.llama:
                        response = get_llama_response(system_prompt, user_prompt, pipe_line, tokenizer)
                    else:
                        response = get_gpt_response(system_prompt, user_prompt, api_key, user, model)
                    
                    print("response: ", response)

                    if "image_file" in json.loads(answers[i]):
                        text = f"Image file-{json.loads(answers[i])['image_file']} "
                    else:
                        text = ""
                    results.append(text+response)

    results = "\n".join(results)
    # save results      
    with open(args.step3_result_path, 'w') as write_file:
        write_file.write(results)