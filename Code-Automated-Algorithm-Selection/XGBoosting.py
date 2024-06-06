import torch
import xgboost

from utils import build_dataset_for_AAS


def execute(data, labels):
    params = {
        "subsample": 0.5,
        "max_depth": 3
        # "sampling_method": "gradient_based",
        # "tree_method": "hist",
        # "device": "cuda"
    }
    data_train = data[:900]
    data_test = data[900:]
    labels_train = labels[:900]
    labels_test = labels[900:]
    algo = xgboost.XGBClassifier(**params)
    algo.fit(data_train, labels_train)
    labels_predicted_probs = algo.predict(data_test)
    correct = 0
    for index, instance in enumerate(labels_predicted_probs):
        labels_test[index] = torch.Tensor(labels_test[index])
        instance = torch.Tensor(instance)
        optimal = (labels_test[index] == torch.max(labels_test[index])).nonzero()
        if instance.argmax(dim=0) in optimal:
            correct += 1

    correct_t = 0
    labels_predicted_probs = algo.predict(data_train)
    for index, instance in enumerate(labels_predicted_probs):
        labels_train[index] = torch.Tensor(labels_train[index])
        instance = torch.Tensor(instance)
        optimal = (labels_train[index] == torch.max(labels_train[index])).nonzero()
        if instance.argmax(dim=0) in optimal:
            correct_t += 1
    return correct_t / len(labels_train), correct / len(labels_test)


dataset = build_dataset_for_AAS('')[0]
dataset_f = [instance[0] for instance in dataset]
dataset_l = [instance[1] for instance in dataset]
print(execute(dataset_f, dataset_l))
