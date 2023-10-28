import sys, os
import shutil
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv()
load_dotenv(env_path)
home_path = os.getenv("HOME_PATH")
sys.path.append(home_path+"/ICTC")

import argparse
import torch
import json
from tqdm import tqdm
import shortuuid

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils.argument import args

from PIL import Image
import math
import re


def read_file_to_string(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content


def load_image_paths_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") \
                or filename.endswith(".JPEG"):  # Add more file types if needed
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
    return image_paths


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def post_process():
    answers = []
    with open(args.clustering_result_path, 'r') as file:
        prev_answers = file.readlines()
        for answer in prev_answers:
            answers.append(answer.split(":")[1].strip())
    return answers


def eval_model(args):
    # Open image folder
    image_files = load_image_paths_from_folder(args.image_folder)
    # image_files = image_files[8743:]
    answers_file = os.path.expanduser(args.step1_result_path)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    qs = read_file_to_string(f"{args.exp_path}/step1_prompt.txt")
    
    cur_prompt = qs
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    for i in tqdm(range(len(image_files))):
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        image_file = image_files[i]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=100,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            print('stop str')
            outputs = outputs[:-len(stop_str)]
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"text": outputs,
                                "image_file": image_file,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()
    if not os.path.exists(f"{args.exp_path}/step1_result.jsonl"):
        shutil.copy(args.answers_file_classification, f"{args.exp_path}/step1_result.jsonl")

if __name__ == "__main__":
    eval_model(args)