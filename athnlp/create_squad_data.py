import copy
import json
import os
from argparse import ArgumentParser
from operator import itemgetter

import numpy as np

parser = ArgumentParser()

parser.add_argument("-dataset_file", default="data/squad/dev-v2.0.json",
                    help="Path to the Devset of SQuAD 2.0", type=str)
parser.add_argument("-output_dir", default="data/squad",
                    help="Output folder were the files train.json and test.json will be saved")
parser.add_argument("--percentage_train",
                    type=float,
                    help="Percentage of questions associated to a given paragraph to retain for training", default=70.0)
parser.add_argument("--wiki_title", help="Wikipedia page title used as reference to create the dataset",
                    default="Normans")
parser.add_argument("--remove_impossible", action='store_true')


def create_dataset_splits(dataset, percentage_train, remove_impossible=True):
    train = {'version': dataset['version'], 'data': []}
    test = {'version': dataset['version'], 'data': []}

    for example_set in dataset["data"]:
        curr_train = {"title": example_set["title"], "paragraphs": []}
        curr_test = {"title": example_set["title"], "paragraphs": []}
        for paragraph in example_set["paragraphs"]:
            num_questions = len(paragraph["qas"])

            question_ids = np.arange(num_questions)

            np.random.shuffle(question_ids)

            ref_index = int(percentage_train * num_questions)
            train_indexes = question_ids[:ref_index]
            test_indexes = question_ids[ref_index:]

            train_paragraph = copy.copy(paragraph)
            train_qas = itemgetter(*train_indexes)(paragraph["qas"])
            if isinstance(train_qas, dict):
                train_qas = [train_qas]
            if remove_impossible:
                train_qas = [x for x in train_qas if not x['is_impossible']]
            train_paragraph["qas"] = train_qas
            test_paragraph = copy.copy(paragraph)
            test_qas = itemgetter(*test_indexes)(paragraph["qas"])
            if isinstance(test_qas, dict):
                test_qas = [test_qas]
            if remove_impossible:
                test_qas = [x for x in test_qas if not x['is_impossible']]
            test_paragraph["qas"] = test_qas

            curr_train["paragraphs"].append(train_paragraph)
            curr_test["paragraphs"].append(test_paragraph)

        train["data"].append(curr_train)
        test["data"].append(curr_test)

    return train, test


def main(args):
    with open(args.dataset_file) as in_file:
        dataset = json.load(in_file)

    # We extract only data associated to the Wikipedia page of the Normans
    filtered_dataset = {'version': dataset['version'], 'data': []}

    for example in dataset["data"]:
        if example["title"] == args.wiki_title:
            filtered_dataset["data"].append(example)

    total_num_paragraphs = 0
    total_num_questions = 0

    for example_set in filtered_dataset["data"]:
        total_num_paragraphs += len(example_set["paragraphs"])

        for paragraph in example_set["paragraphs"]:
            total_num_questions += len(paragraph["qas"])

    print("Wikipedia page title: {}".format(args.wiki_title))
    print("Total number of paragraphs: {}".format(total_num_paragraphs))
    print("Total number of questions: {}".format(total_num_questions))

    train, test = create_dataset_splits(filtered_dataset, args.percentage_train / 100, args.remove_impossible)

    print("-- Saving training and test files to directory: {}".format(args.output_dir))
    with open(os.path.join(args.output_dir, "train.json"), mode="w") as out_file:
        json.dump(train, out_file)

    with open(os.path.join(args.output_dir, "test.json"), mode="w") as out_file:
        json.dump(test, out_file)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
