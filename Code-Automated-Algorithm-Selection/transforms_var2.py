from typing import Union, List, Tuple

import cv2
import torch
import torchvision
from torch import Tensor
from torchvision.transforms import v2


class RandomAugmentation:
    def __init__(self, instances: Union[str, List[int]] = 'all',
                 num_ops: int = 2, magnitude: int = 9):
        self.instances = instances
        self.num_ops = num_ops
        self.magnitude = magnitude

    def __call__(self, sample_par: Union[Tensor, List[Tensor], Tuple[Tensor]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = v2.RandAugment(self.num_ops, self.magnitude)(sample[instance])
        else:
            sample = v2.RandAugment(self.num_ops, self.magnitude)(sample)
        return sample


class ToGrayscale:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor], Tuple[Tensor]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = v2.Grayscale()(sample[instance])
        else:
            sample = v2.Grayscale()(sample)
        return sample


class Flatten:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor], Tuple[Tensor]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = torch.flatten(sample[instance])
        else:
            sample = torch.flatten(sample)
        return sample


class MinMaxNormalization:
    def __init__(self, instances: Union[str, List[int]] = 'all', minmax: bool = False,
                 minim=0, maxim=255):
        self.instances = instances
        self.maxim = maxim
        self.minim = minim
        self.minmax = minmax

    def __call__(self, sample_par: Union[Tensor, List[Tensor], Tuple[Tensor]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            if self.minmax:
                self.minim = sample[self.instances].min()
                self.maxim = sample[self.instances].max()
            for instance in self.instances:
                sample[instance] -= self.minim
                sample[instance] /= (self.maxim - self.minim)
        else:
            if self.minmax:
                self.minim = sample.min()
                self.maxim = sample.max()
            sample -= self.minim
            sample /= (self.maxim - self.minim)
        return sample


class StandardNormalization:
    def __init__(self, mean, std):  # , mean_l, std_l):
        self.mean = mean  # number of features
        self.std = std
        # self.mean_l = mean_l
        # self.std_l = std_l

    def __call__(self, instances):
        instance = torch.Tensor(instances[0])
        instance_l = torch.Tensor(instances[1])
        for index in range(len(self.mean)):
            instance[index] -= self.mean[index]
            instance[index] /= self.std[index]
        # for index in range(len(self.mean_l)):
        #     instance_l[index] -= self.mean_l[index]
        #     instance_l[index] /= self.std_l[index]
        return instance, instances[1]


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, instances):
        return torch.Tensor(instances[0]), torch.Tensor(instances[1])


class Unsqueeze:
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, instances):
        return instances[0].reshape(self.dim1, self.dim2).unsqueeze(0), instances[1]


class ChangeType:
    def __init__(self, dtype, instances: Union[str, List[int]] = 'all'):
        self.instances = instances
        self.desired_type = dtype

    def __call__(self, sample: Union[Tensor, List[Tensor]]):
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = (v2.ToDtype(dtype=self.desired_type, scale=True)
                                    (sample[instance]))
        else:
            sample = (v2.ToDtype(dtype=self.desired_type, scale=True)(sample))
        return sample


class Crop:
    def __init__(self, shape: torch.Size, instances: Union[str, List[int]] = 'all'):
        self.instances = instances
        self.shape = shape

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            # sample[self.instances[0]] = (v2.RandomResizedCrop(self.shape)
            #                              (sample[self.instances[0]]))
            # state = torch.get_rng_state()
            batch = []
            for instance in self.instances:
                # torch.set_rng_state(state)
                batch.append(sample[instance])
            new_images = v2.RandomResizedCrop(
                self.shape, antialias=True)(batch)
            for instance in self.instances:
                sample[instance] = new_images[instance]
        else:
            sample = v2.RandomResizedCrop(self.shape, antialias=True)(sample)
        return sample


class RandomRotation:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            # sample[self.instances[0]] = (v2.RandomRotation(degrees=10)(sample[instance[0]]))
            # state = torch.get_rng_state()
            batch = []
            for instance in self.instances:
                # torch.set_rng_state(state)
                batch.append(sample[instance])
            new_images = v2.RandomRotation(degrees=5)(batch)
            for instance in self.instances:
                sample[instance] = new_images[instance]
        else:
            sample = v2.RandomRotation(degrees=5)(sample)
        return sample


class ColorChange:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor], List[List[Tensor]]]):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            batch = []
            for instance in self.instances:
                batch.append(sample[instance])
            new_images = v2.ColorJitter(brightness=0.5, contrast=1,
                                        saturation=0.5, hue=0.5)(batch)
            for instance in self.instances:
                sample[instance] = new_images[instance]
        else:
            sample = v2.ColorJitter(brightness=0.5, contrast=1,
                                    saturation=0.1, hue=0.5)(sample)
        return sample


class ImageToTensor:
    def __init__(self, instances: Union[str, List[int]] = 'all',
                 device=torch.device('cpu')):
        self.instances = instances
        self.device = device

    def __call__(self, sample_par: Union[str, List[str]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = torch.from_numpy(cv2.imread(
                    sample[instance], cv2.IMREAD_COLOR)).to(self.device)
        else:
            sample = torch.from_numpy(cv2.imread(sample, cv2.IMREAD_COLOR)).to(self.device)
        return sample


class NumberToTensor:
    def __init__(self, instances: Union[str, List[int]] = 'all',
                 device=torch.device('cpu')):
        self.instances = instances
        self.device = device

    def __call__(self, sample_par: Union[int, List[int]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = Tensor([sample[instance]]).to(self.device)
        else:
            sample = Tensor(sample).to(self.device)
        return sample


class FeatureLabelsSplit:
    def __init__(self, features: List[int], labels: List[int], device=torch.device('cpu')):
        self.features = features
        self.labels = labels
        self.device = device

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        features_tensor = Tensor(sample[self.features[0]]).to(self.device)
        labels_tensor = Tensor(sample[self.labels[0]]).to(self.device)
        for instance in self.features[1:]:
            features_tensor = torch.cat((features_tensor, sample[instance]))
        for instance in self.labels[1:]:
            labels_tensor = torch.stack((labels_tensor, sample[instance]))
        return [features_tensor, labels_tensor]


class GroupTensors:
    def __init__(self, instances: Union[str, List[int]] = 'all', position: int = 0,
                 device=torch.device('cpu')):
        self.instances = instances
        self.position = position
        self.device = device

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if self.instances == 'all':
            self.instances = range(len(sample))
        result = Tensor(sample[self.instances[0]]).to(self.device)
        for instance in self.instances[1:]:
            result = torch.cat((result, sample[instance]), dim=0)
        deleted = 0
        for instance in self.instances:
            del sample[instance - deleted]
            deleted += 1
        sample.insert(self.position, result)
        return sample


class ReshapeTensors:
    def __init__(self, shape: Union[Tuple, int],
                 instances: Union[str, List[int]] = 'all'):
        self.instances = instances
        self.shape = shape

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = sample[instance].reshape(self.shape)
        else:
            sample = sample.reshape(self.shape)
        return sample


class UngroupTensors:
    def __init__(self, instance: int, dim: int):
        self.instance = instance
        self.position = 0
        self.dim = dim

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        result = list(sample[self.instance].split(self.dim))
        sample = sample[:self.instance] + result + sample[(self.instance + 1):]
        return sample


class DecomposeChannels:
    def __init__(self, instances: Union[str, List[int]] = 'all',
                 device=torch.device('cpu')):
        self.instances = instances
        self.device = device

    def __call__(self, sample_par):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = Tensor(torch.stack([sample[instance][:, :, 0],
                                                       sample[instance][:, :, 1],
                                                       sample[instance][:, :, 2]])).to(self.device)
        else:
            sample = Tensor(torch.stack([sample[:, :, 0],
                                         sample[:, :, 1],
                                         sample[:, :, 2]])).to(self.device)
        return sample


class RecomposeChannels:
    def __init__(self, instances: Union[str, List[List]] = 'all',
                 device=torch.device('cpu')):
        self.instances = instances
        self.device = device

    def __call__(self, sample_par):
        if type(sample_par) in [list, tuple]:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = [range(x, x + 3) for x in range(0, len(sample), 3)]
            for instance in self.instances:
                sample[instance] = Tensor(torch.stack([sample[instance][0, :, :],
                                                       sample[instance][1, :, :],
                                                       sample[instance][2, :, :]], dim=2)).to(self.device)
        else:
            sample = Tensor(torch.stack([sample[0, :, :],
                                         sample[1, :, :],
                                         sample[2, :, :]], dim=2)).to(self.device)
        return sample


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
