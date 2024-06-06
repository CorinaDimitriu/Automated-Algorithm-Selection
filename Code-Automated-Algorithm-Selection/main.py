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
    runner = Runner(30, utils.get_default_device(),
                    f'.', utils.build_dataset_for_AAS,
                    writer, similarity_func=utils.count_correct_predictions,
                    )
    model = Model(optimizers=[torch.optim.Adam],
                  optimizer_args=[{'lr': 0.001, 'weight_decay': 1e-4,
                                   }],
                  closure=[False],
                  loss=torch.nn.MSELoss(),
                  device=utils.get_default_device(),
                  # lr_scheduler=MultiStepLR,
                  # lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
                  #                    'gamma': 0.1},
                  layers=[
                      torch.nn.Linear(16, 100),
                      torch.nn.ReLU(),
                      torch.nn.Linear(100, 6),
                      torch.nn.Softmax(dim=1)
                  ],
                  to_initialize=[0, 2],
                  weight_initialization=[torch.nn.init.xavier_normal_,
                                         torch.nn.init.xavier_normal_,
                                         ]
                  )

    # in_channels = 1
    # hidden_channels = 2
    # model = Model(optimizers=[torch.optim.Adam],
    #               optimizer_args=[{'lr': 0.0001, 'weight_decay': 1e-4,
    #                                }],
    #               closure=[False],
    #               loss=torch.nn.MSELoss(),
    #               device=utils.get_default_device(),
    #               # lr_scheduler=MultiStepLR,
    #               # lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
    #               #                    'gamma': 0.1},
    #               layers=[
    #                   torch.nn.ConvTranspose2d(in_channels, hidden_channels * 2,
    #                                            kernel_size=(4, 4), stride=1,
    #                                            padding=0, output_padding=0),
    #                   Upsampling(hidden_channels * 2, hidden_channels * 4, dropout=True,
    #                              kernel_size=(4, 4), stride=1,
    #                              padding=0),
    #                   Upsampling(hidden_channels * 4, hidden_channels * 8, dropout=True,
    #                              kernel_size=(4, 4), stride=2,
    #                              padding=0),
    #                   Upsampling(hidden_channels * 8, hidden_channels * 16, dropout=True,
    #                              kernel_size=(4, 4), stride=2,
    #                              padding=1),
    #
    #                   Downsampling(hidden_channels * 16, hidden_channels * 8,
    #                                kernel_size=(4, 4), stride=2,
    #                                padding=1),
    #                   Downsampling(hidden_channels * 8, hidden_channels * 4,
    #                                kernel_size=(4, 4), stride=2,
    #                                padding=0),
    #                   Downsampling(hidden_channels * 4, hidden_channels * 2,
    #                                kernel_size=(4, 4), stride=1,
    #                                padding=0),
    #                   Downsampling(hidden_channels * 2, hidden_channels * 1,
    #                                kernel_size=(4, 4), stride=1,
    #                                padding=0, norm=False),
    #                   torch.nn.Flatten(),
    #                   torch.nn.Linear(32, 100),
    #                   torch.nn.Linear(100, 6),
    #                   torch.nn.Softmax(dim=1)
    #               ],
    #               to_initialize=[0],
    #               weight_initialization=[torch.nn.init.xavier_normal_,
    #                                      ]
    #               )

    runner.run_model(model, transforms=[
        ToTensor(),
        StandardNormalization([57.31786088703234, 896.3902586759368, 1.070247992010226, 3.3241413909468096,
                               292.1816666666667, 300.7038888888889, 300.7038888888889, 327.72786811025367,
                               289.5672222222222, 300.7038888888889, 419863.1448267745, 520974.0930094834,
                               265215.7830517877, 895.7222222222222, 842.2211111111111, 690.6577777777778]
                              ,
                              [92.31514320430337, 6852.967743687869, 0.22483168791339728, 2.14995190243889,
                               660.140052941209, 720.7877618245108, 720.7877618245108, 821.4562598742866,
                               647.1513434386939, 720.7877618245108, 2324574.1082743797, 3177189.4530006787,
                               1721667.3772145363, 2970.870727685283, 2767.2834836510074, 1971.2620133067478]
                              ),
        # StandardNormalization([58.735962939522636, 960.0346192630341, 1.0661863871247814, 3.261964111988714,
        #                        262.2422222222222, 268.4277777777778, 268.4277777777778, 290.8397929427643,
        #                        261.39444444444445, 268.4277777777778, 315089.49642604013, 387930.76028568274,
        #                        202496.26171599567, 765.0511111111111, 722.6188888888889, 611.0588888888889],
        #                       [91.04743961832143, 6910.0163597638075, 0.21178987325853055, 2.067080611612229,
        #                        578.1463268215789, 623.9009014129973, 623.9009014129973, 710.9110892372145,
        #                        568.9444705106918, 623.9009014129973, 2003060.107891521, 2725080.5863040485,
        #                        1523244.761693496, 2551.0592892733116, 2367.617171635015, 1730.8635776137519],
        #                       [5289.402485659656, 5537.0344168260035, 7299.280114722754, 5102.578393881453,
        #                        6678.295411089866, 5974.192160611855],
        #                       [8735.441408639845, 9416.490123067151, 12413.337944809688, 8483.254359658284,
        #                        15650.22909635424, 10025.873322554015]
        #                       ),
        # Unsqueeze(4, 4)
    ],
                     transforms_test=[
                         ToTensor(),
                         StandardNormalization(
                             [57.31786088703234, 896.3902586759368, 1.070247992010226, 3.3241413909468096,
                              292.1816666666667, 300.7038888888889, 300.7038888888889, 327.72786811025367,
                              289.5672222222222, 300.7038888888889, 419863.1448267745, 520974.0930094834,
                              265215.7830517877, 895.7222222222222, 842.2211111111111, 690.6577777777778]
                             ,
                             [92.31514320430337, 6852.967743687869, 0.22483168791339728, 2.14995190243889,
                              660.140052941209, 720.7877618245108, 720.7877618245108, 821.4562598742866,
                              647.1513434386939, 720.7877618245108, 2324574.1082743797, 3177189.4530006787,
                              1721667.3772145363, 2970.870727685283, 2767.2834836510074, 1971.2620133067478]
                             ),
                         # StandardNormalization(
                         #     [58.735962939522636, 960.0346192630341, 1.0661863871247814, 3.261964111988714,
                         #      262.2422222222222, 268.4277777777778, 268.4277777777778, 290.8397929427643,
                         #      261.39444444444445, 268.4277777777778, 315089.49642604013, 387930.76028568274,
                         #      202496.26171599567, 765.0511111111111, 722.6188888888889, 611.0588888888889],
                         #     [91.04743961832143, 6910.0163597638075, 0.21178987325853055, 2.067080611612229,
                         #      578.1463268215789, 623.9009014129973, 623.9009014129973, 710.9110892372145,
                         #      568.9444705106918, 623.9009014129973, 2003060.107891521, 2725080.5863040485,
                         #      1523244.761693496, 2551.0592892733116, 2367.617171635015, 1730.8635776137519],
                         #     [5289.402485659656, 5537.0344168260035, 7299.280114722754, 5102.578393881453,
                         #      6678.295411089866, 5974.192160611855],
                         #     [8735.441408639845, 9416.490123067151, 12413.337944809688, 8483.254359658284,
                         #      15650.22909635424, 10025.873322554015]
                         #     ),
                         # Unsqueeze(4, 4)
                     ],
                     pin_memory=True,
                     transforms_not_cached=[
                     ], batch_size=16, num_workers=4, num_classes=6,
                     val_batch_size=8)
