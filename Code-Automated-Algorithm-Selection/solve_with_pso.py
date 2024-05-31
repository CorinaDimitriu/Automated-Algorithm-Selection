import os


def solve(configuration):  # in this case configuration is the full path to the file's name
    script = ("D:/Facultate/Semestrul_2_Master/AEA/"
              "Code-Automated-Algorithm-Selection/pso/Cpp_Project/x64/Debug/Cpp_Project.exe")
    result = os.popen(" ".join([script, configuration.replace('\\', '/')])).read()
    print("Done")
    return int(result.split("Fitness of best solution: ")[1].split('\n')[0])
