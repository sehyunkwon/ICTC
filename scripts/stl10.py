import os, sys

from dotenv import load_dotenv, find_dotenv
load_dotenv("../cluster_fm/.env")
home_path = os.getenv("HOME_PATH")
sys.path.append(home_path+"/ClusterFM")

# Get the current directory
direc = f'{home_path}/data/stl10/test'

# Function to rename the files in a folder
def rename_files_in_folder(folder_path, label):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.png'):
                old_path = os.path.join(root, filename)
                new_filename = f"{os.path.splitext(filename)[0]}_{label}.png"
                new_path = os.path.join(direc, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} to {new_path}")



# Iterate through subdirectories (assuming each folder corresponds to a label)
for label in os.listdir(direc):
    folder_path = os.path.join(direc, label)
    if os.path.isdir(folder_path):
        rename_files_in_folder(folder_path, label)