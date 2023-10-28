import sys, os
import shutil
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv()
load_dotenv(env_path)
home_path = os.getenv("HOME_PATH")
sys.path.append(home_path+"/ICTC")

import argparse
import json
from tqdm import tqdm
import shortuuid

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from transformers import Blip2Processor, Blip2ForConditionalGeneration

from utils.argument import args

from PIL import Image
import math
import re


def load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto", torch_dtype=torch.float16).cuda()
    return model, processor

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


def eval_model(args):
    # Open image folder
    image_files = load_image_paths_from_folder(args.image_folder)
    answers_file = os.path.expanduser(args.step1_result_path)

    # Model
    model, processor = load_model()

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    qs = read_file_to_string(f"{args.exp_path}/step1_prompt.txt")

    for i in tqdm(range(len(image_files))):
        image_file = image_files[i]
        image = Image.open(os.path.join(args.image_folder, image_file))
        inputs = processor(image, qs, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        out = model.generate(**inputs, max_new_tokens=50)
        outputs = processor.decode(out[0], skip_special_tokens=True)
        print(outputs)


        ans_file.write(json.dumps({"text": outputs,
                                "image_file": image_file,
                                "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()
    if not os.path.exists(f"{args.exp_path}/step1_result.jsonl"):
        shutil.copy(args.step1_result_path, f"{args.exp_path}/step1_result.jsonl")

if __name__ == "__main__":
    eval_model(args)
