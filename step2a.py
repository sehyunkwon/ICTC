import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from utils.argument import args 
from utils.llm_utils import get_gpt_response, get_llama_response

if __name__ == "__main__":
    load_dotenv()
    if args.use_gpt4:
        api_key = os.getenv("API_KEY")
        user = os.getenv("USER")
        model = os.getenv("MODEL")
    else:
        api_key = os.getenv("API_KEY_3.5")
        user = os.getenv("USER_3.5")
        model = os.getenv("MODEL_3.5")

    url = ""
    if args.llama_ver == "llama_70b":
        url = os.getenv("LLAMA_70b_URL")
    elif args.llama_ver == "llama_13b":   
        url = os.getenv("LLAMA_13b_URL")
    elif args.llama_ver == "llama_7b":
        url = os.getenv("LLAMA_7b_URL")
    
    

    results = []
    # read system prompt
    with open(args.step2a_prompt_path, "r") as label_file:
        system_prompt = label_file.read()
    
        # read initial_answer.jsonl
        with open(args.step1_result_path, "r") as answer_file:
            answers = answer_file.readlines()
            for i in tqdm(range(len(answers))):
                user_prompt = json.loads(answers[i])["text"]

                ### Get GPT result ###
                if args.llama:
                    response = get_llama_response(system_prompt, user_prompt, url)
                else:
                    response = get_gpt_response(system_prompt, user_prompt, api_key, user, model)
                    
                if "image_file" in json.loads(answers[i]):
                    text = f"Image file-{json.loads(answers[i])['image_file']}; "
                else:
                    text = ""
                results.append(text+response)
                # print(result)

    ### Save GPT result ###
    results = "\n".join(results)

    # Open the file for writing and write the string to the file
    with open(args.step2a_result_path, 'w') as file:
        file.write(results)
