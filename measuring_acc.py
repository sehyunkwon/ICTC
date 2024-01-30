import os
import re
import torch
import random
from tqdm import tqdm

from utils.argument import args
from utils.hungarian_match import hungarian_evaluate, confusion_matrix

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')


def find_most_similar(word, word_list):
    best_word = None
    highest_similarity = -1

    # Ensure the word has synsets
    word_synsets = wn.synsets(word)
    if not word_synsets:
        return random.choice(word_list)

    for w in word_list:
        for syn in wn.synsets(w):
            for word_syn in word_synsets:
                similarity = word_syn.path_similarity(syn)
                if similarity and similarity > highest_similarity:
                    highest_similarity = similarity
                    best_word = w

    return best_word


def extract_class(file_name):
    match = re.search(r'(\d+)_(.+)\.jpg', file_name)
    if match:
        return match.group(2)
    else:
        match = re.search(r'(\d+)_(.+)\.png', file_name)
        if match:
            return match.group(2)
        return None


if __name__ == "__main__":
    confusion_matrix_save_path = f"{args.exp_path}/confusion_matrix.pdf"
    file_path = args.step3_result_path
    true_classes = []

    with open(file_path, 'r') as file:
        for line in file:
            if '.png' in line:
                before_png = line.split('.png')[0]
                name = before_png.split('_')[-1]
                true_classes.append(name)
            else:
                before_jpg = line.split('.jpg')[0]
                name = before_jpg.split('_')[-1]
                true_classes.append(name)

    unique_elements = list(set(true_classes))

    # load final answers and classes
    final_classes = []
    final_answers = []
    with open(args.step3_result_path, 'r') as result_file:
        file_read = result_file.readlines()
        # post_process
        for answer in file_read:
            try:
                answer = answer.split(":")[1].strip().lower()
                answer = answer.replace(",", "").replace("'", "").replace(".", "")
                final_answers.append(answer)
            except:
                final_answers.append(answer)

    with open(args.step2b_result_path, 'r') as result_file:
        file_read = result_file.readlines()
        for label in file_read:
            if "Reason" not in label and label.strip() != "" and ":" in label:
                final_classes.append(label.split(":")[1].strip().lower())

    final_answers_ = []

    for i in tqdm(range(len(final_answers))):
        if final_answers[i] in final_classes:
            final_answers_.append(final_answers[i])
        else:
            if args.llama:
                most_similar_word = find_most_similar(final_answers[i], final_classes)
            else:
                most_similar_word = random.choice(final_classes)
            final_answers_.append(most_similar_word)

    final_answers = final_answers_

    """
    Hungarian matching (Assignment problem)
    """
    unique_elements = list(set(final_answers))
    element_to_number = {element: i for i, element in enumerate(unique_elements)}
    final_answers_number = torch.tensor([element_to_number[element] for element in final_answers])

    unique_elements = list(set(true_classes))
    element_to_number = {element: i for i, element in enumerate(unique_elements)}
    true_classes_number = torch.tensor([element_to_number[element] for element in true_classes])
    clustering_stats = hungarian_evaluate(targets=true_classes_number, predictions=final_answers_number,
                                class_names=list(set(true_classes)), 
                                compute_confusion_matrix=True, 
                                confusion_matrix_file=confusion_matrix_save_path)
    print(clustering_stats)

    # save clustering_stats
    stats_path = args.exp_path + "/accuracy.txt"
    with open(stats_path, 'w') as f:
        f.write(str(clustering_stats))