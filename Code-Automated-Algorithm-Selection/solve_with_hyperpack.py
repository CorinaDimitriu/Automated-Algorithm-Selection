from hyperpack import HyperPack
from hyperpack.heuristics import BasePackingProblem


def solve(configuration):
    items = dict()
    total_width = 0.0
    for index, item in enumerate(configuration['boxes']):
        items[str(index)] = {'w': item[0], 'l': item[1]}
        total_width += item[0]
    BasePackingProblem.MAX_W_L_RATIO = int(total_width / configuration['width']) + 1

    problem = HyperPack(items=items, strip_pack_width=configuration['width'])
    problem.local_search()
    local_search = int(problem.log_solution().split("[max height] : ")[1].split("\n")[0])

    problem = HyperPack(items=items, strip_pack_width=configuration['width'])
    problem.hypersearch()
    hypersearch = int(problem.log_solution().split("[max height] : ")[1].split("\n")[0])

    print("Done")
    return [local_search, hypersearch]
