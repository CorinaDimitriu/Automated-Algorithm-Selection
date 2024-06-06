import csv
import itertools
import math
import random
import shutil
import os
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_abs_time_difference(date1: Tuple[int, int], date2: Tuple[int, int]) -> int:
    if date1[0] < date2[0]:
        return (12 - date1[1]) + (12 * (date2[0] - date1[0] - 1)) + date2[1]
    elif date1[0] == date2[0] and date1[1] <= date2[1]:
        return date2[1] - date1[1]
    elif date1[0] > date2[0]:
        return (12 - date2[1]) + (12 * (date1[0] - date2[0] - 1)) + date1[1]
    elif date1[0] == date2[0] and date1[1] > date2[1]:  # else
        return date1[1] - date2[1]


def split_dataset(dataset, split_root):  # build new folder structure
    if os.path.exists(split_root):
        return
    os.mkdir(split_root)
    os.mkdir(os.path.join(split_root, 'train'))
    os.mkdir(os.path.join(split_root, 'test'))
    os.mkdir(os.path.join(split_root, 'validate'))
    x_train, x_test = train_test_split(dataset, train_size=0.7, random_state=42)
    x_test, x_validation = train_test_split(x_test, train_size=0.5, random_state=42)

    training_path = os.path.join(split_root, 'train')
    for instance in x_train:
        generate_files(instance, training_path)
    testing_path = os.path.join(split_root, 'test')
    for instance in x_test:
        generate_files(instance, testing_path)
    validation_path = os.path.join(split_root, 'validate')
    for instance in x_validation:
        generate_files(instance, validation_path)


def generate_files(instance, path):
    id_pair = instance[0].split('global_monthly_')[1][:7] + '-' + instance[1].split('global_monthly_')[1][:7]
    split_instance = instance[0][instance[0].find('L15-'):].split('\\')
    new_path = os.path.join(path, split_instance[0], id_pair, split_instance[2])
    try:
        shutil.copyfile(instance[0], new_path)
    except IOError:
        os.makedirs(os.path.dirname(new_path))
        shutil.copyfile(instance[0], new_path)
    split_instance = instance[1][instance[1].find('L15-'):].split('\\')
    new_path = os.path.join(path, split_instance[0], id_pair, split_instance[2])
    shutil.copyfile(instance[1], new_path)


def split_dataset_csv(dataset):
    x_train, x_test = train_test_split(dataset, train_size=0.7, random_state=42)
    x_test, x_validation = train_test_split(x_test, train_size=0.5, random_state=42)

    # csv for training data
    with open('train.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_in', 'image_out', 'time_elapsed'])
        for row in x_train:
            csv_out.writerow(row)
    # csv for testing data
    with open('test.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_in', 'image_out', 'time_elapsed'])
        for row in x_test:
            csv_out.writerow(row)
    # csv for validation data
    with open('validation.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_in', 'image_out', 'time_elapsed'])
        for row in x_validation:
            csv_out.writerow(row)


def build_dataset_initial(dataset_path):
    dataset_size = 0
    photo_pairs = []
    for (root, directories, files) in os.walk(dataset_path):
        photos = []
        for photo in files:
            photo_path = os.path.join(root, photo)
            year_month = os.path.splitext(photo)[0].split('global_monthly_')[1][:7]
            photos.append((photo_path, year_month))
        # build tuples
        pairs = list(itertools.combinations(photos, 2))
        for pair in pairs:
            photo_1 = pair[0][0]
            photo_2 = pair[1][0]
            months = compute_abs_time_difference((int(pair[0][1][:4]), int(pair[0][1][5:])),
                                                 (int(pair[1][1][:4]), int(pair[1][1][5:])))
            photo_pairs.append([photo_1, photo_2, months])
        dataset_size += len(pairs)
    return photo_pairs, dataset_size


def count_correct_predictions(output, labels):
    # correct = 0
    # for index, instance in enumerate(output):
    #     how_many = labels[index].nonzero()[:, 0].shape[0]
    #     if instance.topk(how_many, 0, True, True).indices in labels[index].nonzero()[:, 0]:
    #         correct += 1
    # return correct
    # return (output.argmax(dim=1) == labels).sum().item()  # perfect prediction per instance

    # for results.txt : min instead of max

    correct = 0
    for index, instance in enumerate(output):
        optimal = (labels[index] == torch.max(labels[index])).nonzero()
        if instance.argmax(dim=0) in optimal:
            correct += 1
    return correct


def turn_to_zero(input: Tensor):
    input[:] = 0


def build_dataset_for_AAS(path):
    input_features = '.\\dataset1'
    input_labels = '.\\labels_train.txt'
    dataset = []
    names = []
    all_labels = []
    with open(input_labels, 'r') as file_out:
        for instance in file_out:
            names.append(instance.split(' : ')[0])
            labels = instance.split(' : ')[2].strip('][\n').split(', ')
            # labels = instance.split(' : ')[1].strip('][\n').split(', ')
            labels = [int(label) for label in labels]
            # labels = [int(float(label)) for label in labels]
            all_labels.append(labels)

    # shuffled_indices = list(range(1046))
    # random.shuffle(shuffled_indices)
    # with open("random_r.txt", 'w') as file:
    #     print(shuffled_indices, file=file, flush=True)

    with open("random.txt", 'r') as file:
        data = file.read()
        data = data.strip('][\n').split(', ')
        shuffled_indices = [int(instance) for instance in data]

    for index1 in shuffled_indices:
        name = names[index1]
        labels = all_labels[index1]
        with open(os.path.join(input_features, name), 'r') as file_in:
            configuration = {'boxes': []}
            for index, line in enumerate(file_in):
                line = line.replace('\t', ' ')
                line = line.replace('  ', ' ')
                if index == 0:
                    configuration['width'] = int(line.split(' ')[0])
                elif index == 1:
                    continue
                elif '0' in line or '1' in line or '2' in line or '3' in line or \
                        '4' in line or '5' in line or '6' in line or '7' in line or \
                        '8' in line or '9' in line:
                    width = int(line.split(' ')[0])
                    height = int(line.split(' ')[1].strip())
                    configuration['boxes'].append([width, height])
        # compute effective features to feed NN
        max_aspect_ratio = -math.inf
        max_area_ratio = -math.inf
        mean = 0.0
        mean_width = 0.0
        mean_height = 0.0
        # len_distinct = len(configuration['boxes'])
        counter = dict()
        for i1, box in enumerate(configuration['boxes']):
            aspect_ratio = max(box) / min(box) * 1.0
            max_aspect_ratio = max([max_aspect_ratio, aspect_ratio])
            mean += (box[0] + box[1])
            mean_width += box[0]
            mean_height += box[1]
            counter[tuple(box)] = 1
            for i2, box2 in enumerate(configuration['boxes']):
                if i1 != i2:
                    area_ratio = (box[0] * box[1]) / (box2[0] * box2[1]) * 1.0
                    max_area_ratio = max([max_area_ratio, area_ratio])
                    # if (box[0] == box2[0] and box[1] == box2[1]) or (box[0] == box2[1] and box[1] == box2[0]):
                    #     len_distinct -= 1
        heterogeneity_ratio = len(configuration['boxes']) / len(counter) * 1.0
        mean /= len(configuration['boxes']) * 2
        mean_width /= len(configuration['boxes'])
        mean_height /= len(configuration['boxes'])
        width_ratio = configuration['width'] / mean
        median = np.median(np.array(configuration['boxes']).flatten())
        median_width = np.median(np.array(configuration['boxes'])[:, 0])
        median_height = np.median(np.array(configuration['boxes'])[:, 1])
        variance = np.var(np.array(configuration['boxes']).flatten())
        variance_width = np.var(np.array(configuration['boxes'])[:, 0])
        variance_height = np.var(np.array(configuration['boxes'])[:, 1])
        max_all = np.max(np.array(configuration['boxes']).flatten())
        max_width = np.max(np.array(configuration['boxes'])[:, 0])
        max_height = np.max(np.array(configuration['boxes'])[:, 1])
        features = [max_aspect_ratio, max_area_ratio, heterogeneity_ratio, width_ratio,
                    median, median_width, median_height, mean, mean_width, mean_height,
                    variance, variance_width, variance_height, max_all, max_width, max_height,
                    ]
        labels_3 = 1 if np.array(labels[3:-3]).any() else 0
        # labels_3 = np.min(np.array(labels[3:-3]))
        labels_4 = 1 if np.array(labels[-3:-1]).any() else 0
        # labels_4 = np.min(np.array(labels[-3:-1]))
        new_labels = [labels[0], labels[1], labels[2], labels_3, labels_4, labels[5]]
        dataset.append([features, new_labels])
    return dataset, len(dataset)


# dataset = build_dataset_for_AAS('')[0]
# dataset_f = [instance[0] for instance in dataset]
# dataset_f = np.array(dataset_f, dtype=np.float64)
# means = []
# sd = []
# for features_i in np.array(dataset_f[:900]).T:
#     means.append(np.mean(features_i))
#     sd.append(np.std(features_i))
# print(means)
# print(sd)
#
# labels = [instance[1] for instance in dataset]
# labels = np.array(labels, dtype=np.float64)
# means = []
# sd = []
# for features_i in labels.T:
#     means.append(np.mean(features_i))
#     sd.append(np.std(features_i))
# print(means)
# print(sd)
