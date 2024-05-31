import os
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

dataset1 = './dataset1'
dataset2 = './dataset2'

out = open('labels_train.txt', 'a')
out_res = open('results.txt', 'a')
# index = 0
# for file in os.listdir(dataset1):
#     if file != '3023.in':
#         index += 1
#     else:
#         break
# print(index)
for file in os.listdir(dataset1)[1252:]:
    configuration = {'boxes': []}
    shelf = True
    with (open(os.path.join(dataset1, file), 'r') as file_open):
        for index, line in enumerate(file_open):
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
                if not (width <= configuration['width'] and height <= configuration['width']):
                    shelf = False
                configuration['boxes'].append([width, height])
    if len(configuration['boxes']) <= 100 and shelf:
        results = [solve_with_priority_guillotine.solve(configuration),
                   solve_with_GA.solve(configuration),  #
                   solve_with_steinberg.solve(configuration),
                   *solve_with_gurobi.solve(configuration),  #
                   *solve_with_hyperpack.solve(configuration),  #
                   solve_with_pso.solve(os.path.abspath(os.path.join(dataset1, file))),
                   # solve_with_cp.solve(configuration),
                   # solve_with_lp.solve(configuration),
                   # solve_with_sat.solve(configuration),
                   # solve_with_smt.solve(configuration)
                   ]
        print(" : ".join([file, str(results)]), file=out_res, end='\n', flush=True)
        optimum = min(results)
        for index, value in enumerate(results):
            if value == optimum:
                results[index] = int(1)
            else:
                results[index] = int(0)
        print(" : ".join([file, str(optimum), str(results)]), file=out, end='\n', flush=True)
