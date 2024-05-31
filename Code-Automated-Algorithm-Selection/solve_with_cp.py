# Execute instance "ins-3" with CP "model_6D1"
import os


def solve(configuration):
    transformed_configuration = f"{configuration['width']}\n{len(configuration['boxes'])}\n"
    for index, instance in enumerate(configuration['boxes']):
        transformed_configuration += f"{instance[0]} {instance[1]}\n"
    with open("D:\\Facultate\\Semestrul_2_Master\\AEA\\Code-Automated-Algorithm-Selection\\"
              "vlsi\\instances\\test.txt", "w") as out:
        print(transformed_configuration, file=out, flush=True)
    result = (os.popen(" ".join(["python vlsi/src/scripts/execute_cp.py model_r_7B",
                                 "test", "--no-visualize-output"]))).read()
    print("Done")
    return int(result.split("\'l\': ")[1].split(',')[0])
