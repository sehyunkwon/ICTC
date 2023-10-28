import os
import re
import torch
import random
from argument import args
from hungarian_match import hungarian_evaluate, confusion_matrix

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
    file_names = os.listdir(args.image_folder)
    true_classes = [extract_class(file_name) for file_name in file_names if extract_class(file_name) is not None]
    print(true_classes)

    # load final answers and classes
    final_classes = []
    final_answers = []
    with open(args.classification_result_path, 'r') as result_file:
        file_read = result_file.readlines()
        # post_process
        for answer in file_read:
            try:
                final_answers.append(answer.split(":")[1].strip().lower())
            except:
                final_answers.append(answer)

    with open(args.clustering_result_path, 'r') as result_file:
        file_read = result_file.readlines()
        for label in file_read:
            if "Reason" not in label and label.strip() != "" and ":" in label:
                final_classes.append(label.split(":")[1].strip().lower())

    final_answers_ = []
    wrong_num = 0
    for i in range(len(final_answers)):
        if final_answers[i] in final_classes:
            final_answers_.append(final_answers[i])
        else:
            most_similar_word = random.choice(final_classes)
            final_answers_.append(most_similar_word)
            print(final_answers[i], most_similar_word)
            wrong_num += 1
    final_answers = final_answers_
    print('Wrong Assigned: ', wrong_num)

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