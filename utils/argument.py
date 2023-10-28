import os
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from time import strftime, localtime


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Experiment arguments for ICTC")

#datadir
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10", "ppmi", "stanford-40-actions"])

# Hyperparameters for LLaVa
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default="llava_v1")
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--num_beams", type=int, default=1)

# Hyperparamteres for GPT
parser.add_argument("--gpt_temperature", type=float, default=0.05)
parser.add_argument("--gpt_top_p", type=float, default=0.9)
parser.add_argument("--frequency_penalty", type=float, default=0.5)
parser.add_argument("--presence_penalty", type=float, default=0.5)

# LlaMa-2
parser.add_argument("--llama", type=str2bool, default=False)
parser.add_argument("--llama_ver", type=str, default="llama_7b", choices = ["llama_7b", "llama_13b", "llama_70b"])

# LLaVa only
parser.add_argument("--llava_only", type=str2bool, default=False)

# whether to use gpt4 for labeling process
parser.add_argument("--use_gpt4", type=str2bool, default=False)

# logging directory
parser.add_argument("--exp_name", type=str, default="name_your_experiment")

# num classes for ppmi dataset
parser.add_argument("--num_classes", type=int, default=12, choices=[2,4,7,8], help="4 for stanford-40-actions mood; 8 for location; rest for ppmi")

# categories for stanford-40-actions
parser.add_argument("--cl_criteria", type=str, default="action", choices=["action", "location", "mood"])

args = parser.parse_args()

load_dotenv()
args.home_path = os.getenv("HOME_PATH")

args.data_dir = f"{args.home_path}/ICTC/data/"
args.image_folder = f"{args.home_path}/data/"

# setup dataset and clustering criteria
if args.dataset == "cifar10":
    args.image_folder += "cifar10/"
    args.num_classes = 10
elif args.dataset == "cifar100":
    args.image_folder += "cifar100/"
    args.num_classes = 20
elif args.dataset == "stl10":
    args.image_folder += "stl10/test/"
    args.num_classes = 10
elif args.dataset == "ppmi":
    args.image_folder += "ppmi/"
    if args.num_classes == 12:
        args.image_folder += "12_classes/"
    elif args.num_classes == 7:
        args.image_folder += "7_classes/"
    elif args.num_classes == 2:
        args.image_folder += "2_classes/"
elif args.dataset == "stanford-40-actions":
    args.image_folder += "stanford-40-actions/JPEGImages/"
    if args.cl_criteria == "actions":
        args.num_classes = 40

# Step 1 Datadirs
args.step1_prompt_path = args.data_dir + f"{args.dataset}/step1_prompt"
if args.cl_criteria == "location" and args.dataset == "stanford-40-actions":
    args.step1_prompt_path += "_location.txt"
elif args.cl_criteria == "mood" and args.dataset == "stanford-40-actions":
    args.step1_prompt_path += "_mood.txt"
else:
    args.step1_prompt_path += ".txt"

# Step 2a Datadirs
args.step2a_prompt_path = args.data_dir + f"{args.dataset}/step2a_prompt"
if args.cl_criteria == "location" and args.dataset == "stanford-40-actions":
    args.step2a_prompt_path += "_location.txt"
elif args.cl_criteria == "mood" and args.dataset == "stanford-40-actions":
    args.step2a_prompt_path += "_mood.txt"
else:
    args.step2a_prompt_path += ".txt"

# Step 2b Datadirs
args.step2b_prompt_path = args.data_dir + f"{args.dataset}/step2b_prompt"
if args.cl_criteria == "location" and args.dataset == "stanford-40-actions":
    args.step2b_prompt_path += "_location.txt"
elif args.cl_criteria == "mood" and args.dataset == "stanford-40-actions":
    args.step2b_prompt_path += "_mood.txt"
else:
    args.step2b_prompt_path += ".txt"

# Step 3 Datadirs
args.step3_prompt_path = args.data_dir + f"{args.dataset}/step3_prompt"
if args.cl_criteria == "location" and args.dataset == "stanford-40-actions":
    args.step3_prompt_path += "_location.txt"
elif args.cl_criteria == "mood" and args.dataset == "stanford-40-actions":
    args.step3_prompt_path += "_mood.txt"
else:
    args.step3_prompt_path += ".txt"

hyperparameters = ""
if args.temperature != 0.2:
    hyperparameters += f"_temp_{args.temperature}"
if args.top_p != 0.9:
    hyperparameters += f"_top_p_{args.top_p}"
if args.gpt_top_p != 0.9:
    hyperparameters += f"_gpt_top_p_{args.gpt_top_p}"
if args.gpt_temperature != 0.05:
    hyperparameters += f"_gpt_temp_{args.gpt_temperature}"

hyperparameters.replace('_', '/', 1)

result_dir = args.data_dir + f"results" + hyperparameters + args.exp_name

args.confusion_matrix_save_path = result_dir + "/confusion_matrix.png"

args.exp_path = args.data_dir + args.dataset

if args.llama:
    args.exp_path += f"/{args.llama_ver}/"
elif args.use_gpt4:
    args.exp_path += f"/gpt4/"
else:
    args.exp_path += f"/gpt3.5/"

if args.dataset == "ppmi":
    args.exp_path += f"{args.num_classes}_classes_"

if args.dataset == "stanford-40-actions":
    args.exp_path += f"{args.cl_criteria}_{args.num_classes}_classes_"

args.exp_path += f"{args.exp_name}"

# copy the prompts for record
Path(args.exp_path).mkdir(parents=True, exist_ok=True)

args.step1_result_path = f"{args.exp_path}/step1_result.jsonl"
args.step2a_result_path = f"{args.exp_path}/step2a_result.txt"
args.step2b_result_path = f"{args.exp_path}/step2b_result.txt"
args.step3_result_path = f"{args.exp_path}/step3_result.txt"

if not os.path.exists(f"{args.exp_path}/step1_prompt.txt"):
    shutil.copy(args.step1_prompt_path, f"{args.exp_path}/step1_prompt.txt")
if not os.path.exists(f"{args.exp_path}/step2a_prompt.txt"):
    shutil.copy(args.step2a_prompt_path, f"{args.exp_path}/step2a_prompt.txt")
if not os.path.exists(f"{args.exp_path}/step2b_prompt.txt"):
    shutil.copy(args.step2b_prompt_path, f"{args.exp_path}/step2b_prompt.txt")
if not os.path.exists(f"{args.exp_path}/step3_prompt.txt"):
    shutil.copy(args.step3_prompt_path, f"{args.exp_path}/step3_prompt.txt")