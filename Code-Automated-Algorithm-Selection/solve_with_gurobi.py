import math

from floorplanning.run_all import execute
from floorplanning.find_best import execute_best


def solve(configuration):
    transformed_configuration = f"chipwidth : {configuration['width']}\nnumblocks : {len(configuration['boxes'])}\n"
    for index, instance in enumerate(configuration['boxes']):
        if instance[0] <= configuration['width']:
            transformed_configuration += f"{index} : {instance[0]} {instance[1]}\n"
        else:
            transformed_configuration += f"{index} : {instance[1]} {instance[0]}\n"
    with open("D:\\Facultate\\Semestrul_2_Master\\AEA\\Code-Automated-Algorithm-Selection\\"
              "floorplanning\\testcase\\test.txt", "w") as out:
        print(transformed_configuration, file=out, flush=True)
    print("Done")
    return execute()
    # result = execute_best(len(configuration['boxes']))
    # minim = math.inf
    # for line in result.split('\n'):
    #     sol_number = int(line.split(' ')[-1])
    #     if sol_number < minim:
    #         minim = sol_number
    # return minim
