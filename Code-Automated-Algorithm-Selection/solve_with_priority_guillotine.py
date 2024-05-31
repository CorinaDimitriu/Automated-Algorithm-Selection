from priority_guillotine.spp.ph import phspprg


def solve(configuration):
    width = configuration['width']
    height1, rectangles1 = phspprg(width, configuration['boxes'])
    height2, rectangles2 = phspprg(width, configuration['boxes'], sorting='height')
    print("Done")
    return min(height1, height2)
