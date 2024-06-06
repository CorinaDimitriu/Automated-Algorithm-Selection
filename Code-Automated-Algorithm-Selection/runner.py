import copy
import random

import torch
import torchvision
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchensemble.utils.logging import set_logger
from torchvision import datasets
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import v2

import MLP
from torchensemble import VotingClassifier

from EncoderDecoder import Upsampling, Downsampling
from dataset import Dataset
from loader import TrainLoader, TestLoader, ValidateLoader
from model import Model
from train import TrainTune
from utils import split_dataset, split_dataset_csv


class Runner:  # corresponds to scenario when dataset needs to be loaded and processed from local device, as we work;
    # for including other various scenarios a super class might be built
    def __init__(self, epochs: int, device: torch.device('cpu'),
                 dataset_path, dataset_builder, writer: SummaryWriter,
                 similarity_func=torch.nn.CosineSimilarity(),
                 treshold=0.95):
        self.dataset_builder = dataset_builder
        self.epochs = epochs
        self.device = device
        self.dataset_path = dataset_path
        self.similarity = similarity_func
        self.treshold = treshold
        self.writer = writer
        self.sweep_config = {
            'method': 'random'
        }
        self.configure_sweep()

    def run_model(self, model: Model, split_path='',
                  transforms=None, transforms_test=None, transforms_not_cached=None,
                  pin_memory=False, batch_size=64, val_batch_size=128,
                  num_workers=2, persistent_workers=True,
                  config=None, num_classes=10, model_forward=None):
        # set pin_memory as true if GPU is not used for transformations

        dataset = Dataset(self.dataset_path, self.dataset_builder)
        # train_dataset = [instance[0] for instance in dataset[:200]]
        # test_dataset = [instance[0] for instance in dataset[200:]]
        # if len(transforms) > 0:
        #     for transform in transforms:
        #         train_dataset = transform(train_dataset)
        # dataset[:200][0] = copy.deepcopy(train_dataset)
        # if len(transforms_test) > 0:
        #     for transform in transforms:
        #         test_dataset = transform(test_dataset)
        # dataset[200:][0] = copy.deepcopy(test_dataset)
        dataset = tuple([(tuple(x[0]), tuple(x[1])) for x in dataset])
        train_dataset_ = Dataset(self.dataset_path, lambda path: (tuple([x for x in dataset[:900]]), 900),
                                 transformations=transforms, transformations_test=transforms_test)
        test_dataset_ = Dataset(self.dataset_path, lambda path: (tuple([x for x in dataset[900:]]), 144),
                                transformations=transforms, transformations_test=transforms_test,
                                training=False)
        train_loader = DataLoader(train_dataset_, shuffle=True, pin_memory=pin_memory,
                                  num_workers=num_workers, persistent_workers=persistent_workers,
                                  batch_size=batch_size, drop_last=True)
        validation_loader = DataLoader(test_dataset_, shuffle=False, pin_memory=True, num_workers=0,
                                       batch_size=val_batch_size, drop_last=False)

        # train_tune = TrainTune(model, train_loader, validation_loader, self.writer,
        #                        device=self.device, similarity=self.similarity, treshold=self.treshold,
        #                        config=config, no_class=num_classes, model_forward=model_forward)
        # train_tune.run(self.epochs)

        class MLP1(torch.nn.Module):
            def __init__(self):
                super(MLP1, self).__init__()
                self.layers = [torch.nn.Linear(16, 36),
                               torch.nn.ReLU(),
                               torch.nn.Linear(36, 72),
                               torch.nn.ReLU(),
                               torch.nn.Linear(72, 144),
                               torch.nn.ReLU(),
                               torch.nn.Linear(144, 10),
                               torch.nn.ReLU(),
                               torch.nn.Linear(10, 144),
                               torch.nn.ReLU(),
                               torch.nn.Linear(144, 6),
                               torch.nn.Softmax(dim=1)]
                torch.nn.init.xavier_normal_(self.layers[0].weight)
                torch.nn.init.xavier_normal_(self.layers[2].weight)
                torch.nn.init.xavier_normal_(self.layers[4].weight)
                torch.nn.init.xavier_normal_(self.layers[6].weight)
                torch.nn.init.xavier_normal_(self.layers[8].weight)
                torch.nn.init.xavier_normal_(self.layers[10].weight)
                self.layers = torch.nn.Sequential(*self.layers)

            def forward(self, features: Tensor):
                features = self.layers(features)
                return features

        class Conv1(torch.nn.Module):
            def __init__(self):
                super(Conv1, self).__init__()
                in_channels = 1
                hidden_channels = 2
                self.layers = [torch.nn.ConvTranspose2d(in_channels, hidden_channels * 2,
                                                        kernel_size=(4, 4), stride=1,
                                                        padding=0, output_padding=0),
                               Upsampling(hidden_channels * 2, hidden_channels * 4, dropout=True,
                                          kernel_size=(4, 4), stride=1,
                                          padding=0),
                               Upsampling(hidden_channels * 4, hidden_channels * 8, dropout=True,
                                          kernel_size=(4, 4), stride=2,
                                          padding=0),
                               Upsampling(hidden_channels * 8, hidden_channels * 16, dropout=True,
                                          kernel_size=(4, 4), stride=2,
                                          padding=1),

                               Downsampling(hidden_channels * 16, hidden_channels * 8,
                                            kernel_size=(4, 4), stride=2,
                                            padding=1),
                               Downsampling(hidden_channels * 8, hidden_channels * 4,
                                            kernel_size=(4, 4), stride=2,
                                            padding=0),
                               Downsampling(hidden_channels * 4, hidden_channels * 2,
                                            kernel_size=(4, 4), stride=1,
                                            padding=0),
                               Downsampling(hidden_channels * 2, hidden_channels * 1,
                                            kernel_size=(4, 4), stride=1,
                                            padding=0, norm=False),
                               torch.nn.Flatten(),
                               torch.nn.Linear(32, 100),
                               torch.nn.Linear(100, 6),
                               torch.nn.Softmax(dim=1)]
                torch.nn.init.xavier_normal_(self.layers[0].weight)
                self.layers = torch.nn.Sequential(*self.layers)

            def forward(self, features: Tensor):
                features = self.layers(features)
                return features

        ensemble = VotingClassifier(
            estimator=MLP1,  # here is your deep learning model
            n_estimators=10,  # number of base estimators
            cuda=True
        )
        # Set the criterion
        criterion = torch.nn.MSELoss()  # training objective
        ensemble.set_criterion(criterion)

        # Set the optimizer
        ensemble.set_optimizer(
            "Adam",  # type of parameter optimizer
            lr=1e-3,  # learning rate of parameter optimizer
            weight_decay=1e-4,  # weight decay of parameter optimizer
        )

        # # Set the learning rate scheduler
        ensemble.set_scheduler(
            "CosineAnnealingLR",  # type of learning rate scheduler
            T_max=self.epochs,  # additional arguments on the scheduler
        )

        # Train the ensemble
        ensemble.fit(
            train_loader,
            epochs=self.epochs,  # number of training epochs
            test_loader=validation_loader
        )
        output = ensemble.predict(validation_loader)
        correct = 0
        labels = [x[1] for x in test_dataset_]
        for index, instance in enumerate(output):
            optimal = (labels[index] == torch.max(labels[index])).nonzero()
            if instance.argmax(dim=0) in optimal:
                correct += 1
        print(correct / len(output))

    def configure_sweep(self):
        metric = {
            'name': 'accuracy',
            'goal': 'maximize'
        }
        self.sweep_config['metric'] = metric
        parameters_dict = {
            'batch_size': {
                'values': [32, 64, 128]
            },
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop', 'adagrad', 'sam_sgd']
            },
            'fc_layer_size_1': {
                'values': [4000, 1024, 1000, 512, 256, 128]
            },
            'fc_layer_size_2': {
                'values': [4000, 1024, 1000, 512, 256, 128]
            },
            'fc_layer_size_3': {
                'values': [4000, 1024, 1000, 512, 256, 128]
            },
            'dropout_1': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'dropout_2': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'dropout_3': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'dropout_4': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'batch_norm': {
                'values': [True, False]
            },
            'weight_decay': {
                'values': [0.0, 0.1, 0.001, 0.005, 0.0005, 0.00001]
            },
            'momentum': {
                'values': [0.0, 0.9]
            },
            'Nesterov': {
                'values': [True, False]
            },
            'lr': {
                'values': [0.1, 0.001, 0.005, 0.0005, 0.00001]
            },
            'gradient_clipping': {
                'values': [True, False]
            },
            'lr_scheduler': {
                'values': [True, False]
            },
        }
        self.sweep_config['parameters'] = parameters_dict
        parameters_dict.update({
            'epochs': {
                'value': self.epochs}
        })
