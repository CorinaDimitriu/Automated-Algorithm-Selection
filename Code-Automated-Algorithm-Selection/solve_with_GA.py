from genetic_algorithm.src import application, GLOBAL


def solve(configuration):
    GLOBAL.instances = configuration
    # GLOBAL.RECTANGLES_NUMBER = len(configuration['boxes'])
    return application.execute()
