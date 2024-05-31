from multiprocessing import freeze_support
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import utils
from EncoderDecoder import Upsampling, Downsampling
from model import Model
from runner import Runner

import solve_with_GA
import solve_with_cp
import solve_with_gurobi
import solve_with_hyperpack
import solve_with_lp
import solve_with_priority_guillotine
import solve_with_pso
import solve_with_sat
import solve_with_smt
import solve_with_steinberg
from transforms_var2 import StandardNormalization, ToTensor, Unsqueeze

# configuration = {'boxes': [
#         [5, 3], [5, 3], [2, 4], [30, 8], [10, 20],
#         [20, 10], [5, 5], [5, 5], [10, 10], [10, 5],
#         [6, 4], [1, 10], [8, 4], [6, 6]
#     ],
#     'width': 30}

# configuration = "D:/Facultate/Semestrul_2_Master/AEA/Code-Automated-Algorithm-Selection/pso/Instancias/test.txt"

# print(solve_with_gurobi.solve(configuration))

if __name__ == '__main__':
    freeze_support()
    writer = SummaryWriter()
    runner = Runner(300, utils.get_default_device(),
                    f'.', utils.build_dataset_for_AAS,
                    writer, similarity_func=utils.count_correct_predictions,
                    )
    # model = Model(optimizers=[torch.optim.Adam],
    #               optimizer_args=[{'lr': 0.001, 'weight_decay': 1e-4,
    #                                }],
    #               closure=[False],
    #               loss=torch.nn.MSELoss(),
    #               device=utils.get_default_device(),
    #               # lr_scheduler=MultiStepLR,
    #               # lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
    #               #                    'gamma': 0.1},
    #               layers=[
    #                   torch.nn.Linear(4, 100),
    #                   torch.nn.Linear(100, 6),
    #                   torch.nn.Softmax()
    #               ],
    #               to_initialize=[0],
    #               weight_initialization=[torch.nn.init.xavier_normal_
    #                                      ]
    #               )

    in_channels = 1
    hidden_channels = 4
    model = Model(optimizers=[torch.optim.Adam],
                  optimizer_args=[{'lr': 0.001, 'weight_decay': 1e-4,
                                   }],
                  closure=[False],
                  loss=torch.nn.CrossEntropyLoss(),
                  device=utils.get_default_device(),
                  # lr_scheduler=MultiStepLR,
                  # lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
                  #                    'gamma': 0.1},
                  layers=[
                      torch.nn.ConvTranspose2d(in_channels, hidden_channels * 8,
                                               kernel_size=(4, 1), stride=1,
                                               padding=0, output_padding=0),

                      Upsampling(hidden_channels * 8, hidden_channels * 8, dropout=True,
                                 kernel_size=(4, 1), stride=1,
                                 padding=0),
                      Upsampling(hidden_channels * 8, hidden_channels * 16, dropout=True,
                                 kernel_size=(4, 1), stride=1,
                                 padding=0),
                      Upsampling(hidden_channels * 16, hidden_channels * 16, dropout=True,
                                 kernel_size=(4, 1), stride=1,
                                 padding=0),
                      Upsampling(hidden_channels * 16, hidden_channels * 16,
                                 kernel_size=(16, 1), stride=(2, 1),
                                 padding=(1, 0)),
                      Upsampling(hidden_channels * 16, hidden_channels * 8,
                                 kernel_size=(16, 1), stride=(2, 1),
                                 padding=(1, 0)),
                      Upsampling(hidden_channels * 8, hidden_channels * 4,
                                 kernel_size=(16, 1), stride=(2, 1),
                                 padding=(1, 0)),
                      Upsampling(hidden_channels * 4, hidden_channels * 2,
                                 kernel_size=(16, 1), stride=(2, 1),
                                 padding=(1, 0)),

                      torch.nn.Conv2d(hidden_channels * 2, hidden_channels,
                                      kernel_size=(16, 1), stride=1, padding=1),

                      Downsampling(hidden_channels, hidden_channels, norm=False,
                                   kernel_size=(16, 1), stride=(2, 1),
                                   padding=(1, 0)),
                      Downsampling(hidden_channels, hidden_channels * 2,
                                   kernel_size=(16, 1), stride=(2, 1),
                                   padding=(1, 0)),
                      Downsampling(hidden_channels * 2, hidden_channels * 4,
                                   kernel_size=(16, 1), stride=(2, 1),
                                   padding=(1, 0)),
                      Downsampling(hidden_channels * 4, hidden_channels * 8,
                                   kernel_size=(16, 1), stride=(2, 1),
                                   padding=(1, 0)),
                      Downsampling(hidden_channels * 8, hidden_channels * 8,
                                   kernel_size=(4, 1), stride=1,
                                   padding=0),
                      Downsampling(hidden_channels * 8, hidden_channels * 8,
                                   kernel_size=(4, 1), stride=1,
                                   padding=0),
                      Downsampling(hidden_channels * 8, hidden_channels * 8,
                                   kernel_size=(4, 1), stride=1,
                                   padding=0),
                      Downsampling(hidden_channels * 8, hidden_channels * 8, norm=False,
                                   kernel_size=(4, 1), stride=1,
                                   padding=0),
                      torch.nn.Flatten(),
                      torch.nn.Linear(288, 6),
                      torch.nn.Softmax(dim=1)
                  ],
                  to_initialize=[8],
                  weight_initialization=[torch.nn.init.xavier_normal_,
                                         ]
                  )

    runner.run_model(model, transforms=[
        ToTensor(),
        StandardNormalization([43.28166601975881, 528.9840656837786,
                               1.1186046006936434, 3.5325386381129102],
                              [84.15071540450275, 1709.9547566801778,
                               0.27712741411162967, 2.245307161111377]
                              ),
        Unsqueeze()
    ],
                     transforms_test=[
                         ToTensor(),
                         StandardNormalization([43.28166601975881, 528.9840656837786,
                                                1.1186046006936434, 3.5325386381129102],
                                               [84.15071540450275, 1709.9547566801778,
                                                0.27712741411162967, 2.245307161111377]
                                               ),
                         Unsqueeze()
                     ],
                     pin_memory=True,
                     transforms_not_cached=[
                     ], batch_size=20, num_workers=4, num_classes=6,
                     val_batch_size=20)
