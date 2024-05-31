import copy
from steinberg import Packing
from steinberg.Steinberg import RunSteinberg


def execute(config):
    elements, stripWidth = config['boxes'], config['width']
    # colors = Examples.GenerateColors(len(elements))
    packing, stripHeight = RunSteinberg(stripWidth, copy.deepcopy(elements), False, False)
    first_height = Packing.MaxHeight(packing)
    # Packing.PlotPacking(packing, stripWidth, stripHeight, colors)

    packing, stripHeight = RunSteinberg(stripWidth, copy.deepcopy(elements), True, False)
    second_height = Packing.MaxHeight(packing)
    # Packing.PlotPacking(packing, stripWidth, stripHeight, colors)

    packing, stripHeight = RunSteinberg(stripWidth, copy.deepcopy(elements), False, True)
    third_height = Packing.MaxHeight(packing)
    # Packing.PlotPacking(packing, stripWidth, stripHeight, colors)

    print("Done")
    return min([first_height, second_height, third_height])
