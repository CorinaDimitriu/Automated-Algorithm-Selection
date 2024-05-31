import os


def execute():
    cpp_executable = ("D:/Facultate/Semestrul_2_Master/AEA/Code-Automated-Algorithm-Selection/floorplanning"
                      "/Cpp_Project/x64/Debug/Cpp_Project.exe")

    testcase_home = "D:/Facultate/Semestrul_2_Master/AEA/Code-Automated-Algorithm-Selection/floorplanning/testcase/"
    testcases = ["test.txt"]

    # algorithm:
    ## 0: MILP
    ## 1: shelf w | w/o guillotine
    ## 2: skyline w | w/o guillotine

    # algorithm = 0
    # os.system(" ".join([cpp_executable, testcase_home+"10.txt", str(algorithm), str(10)]))
    # ! Since other MILP cases leads to timeout (>2 hours), please run them separately using command `./floorplan ./testcase/<testcase_file> 0 <you_own_time_out(seconds)>`

    algos = []

    algorithm = 1
    sub_alg_types = range(16)
    sort_types = range(6)
    for testcase in testcases:
        for sub_type in sub_alg_types:
            for sort_type in sort_types:
                res = os.popen(
                    " ".join([cpp_executable, testcase_home + testcase, str(algorithm),
                              str(sub_type), str(sort_type)])).read()
                algos.append(int(res.split("Optimum Height: ")[1].split('\n')[0]))

    algorithm = 2
    sub_alg_types = range(4)
    sort_types = range(6)
    for testcase in testcases:
        for sub_type in sub_alg_types:
            for sort_type in sort_types:
                res = os.popen(
                    " ".join([cpp_executable, testcase_home + testcase, str(algorithm),
                              str(sub_type), str(sort_type)])).read()
                algos.append(int(res.split("Optimum Height: ")[1].split('\n')[0]))
    return algos

