import time
import requests
import json

def setup_gpt(messages, temperature=0.05, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5):
    data = {
        "messages" : messages,
        "temperature" : temperature,
        "top_p": top_p,
        "frequency_penalty" : frequency_penalty,
        "presence_penalty" : presence_penalty
    }
    return data

def get_gpt_response(system_prompt, user_prompt, api_key, user, model):

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        data = setup_gpt(message)
        
        # repeat upto 5 times
        for num_iter in range(5):
            try:
                url = f"https://chatgptapi.krafton-ai.com/{user}/openai/deployments/"
                url += f"{model}/chat/completions?api-version=2023-03-15-preview"
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(data),
                )
                response = response.json()
                if "choices" in response and response["choices"][0]["finish_reason"] == "content_filter":
                    return "ERROR_FILTER"
                elif "error" in response and response['error']['code'] == 'context_length_exceeded':
                    return "ERROR_CONTEXT_LENGTH"
                result = response["choices"][0]["message"]["content"].strip()
                return result
            except Exception as e:
                print(e)
                print(f"Invalid GPT result. Trying again: {num_iter}")
                time.sleep(5 * (2 ** num_iter))
            
        return "ERROR_TIMEOUT"

def setup_llama_prompt(user_prompt, system_prompt):
    prompt = '''<s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message }} [/INST]'''
    prompt = prompt.replace("{{ system_prompt }}", system_prompt)
    prompt = prompt.replace("{{ user_message }}", user_prompt)
    return prompt

def setup_llama(user_prompt, system_prompt, temperature=0.01, top_p=0.9, max_tokens=1024):
    prompt = setup_llama_prompt(user_prompt, system_prompt)
    data = {
        "prompt" : prompt,
        "temperature" : temperature,
        "top_p": top_p,
        "max_tokens" : max_tokens,
    }
    return data

def get_llama_response(system_prompt, user_prompt, url):
    headers = {
        "Content-Type": "application/json",
    }
    data = setup_llama(user_prompt, system_prompt)

    # repeat upto 5 times
    for num_iter in range(5):
        try:
            url = f"https://{url}/generate"
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(data),
                verify=False,
            )
            response = response.json()
            result = str(response["text"]).split("[/INST]")[1].strip(" ").strip("]")
            result = result.replace("'", "")
            return result
        except Exception as e:
            print(e)
            print(f"Invalid LLama result. Trying again: {num_iter}")
            time.sleep(5 * (2 ** num_iter))
        
    return "ERROR_TIMEOUT"

if __name__ == "__main__":
    get_llama_response("You are to classify an image based on the description. Answer in the following format: Answer: {label}", "In the image, two young women are riding camels in the desert. They are sitting on the camels, which are carrying them across the sandy terrain. The women are wearing shorts and sandals, and they appear to be enjoying their ride. The camels are walking in the desert, and the background features a sandy landscape with some vegetation. This scene captures a moment of adventure and exploration in the desert, as the women experience the unique and exotic environment on the back of these animals.",)