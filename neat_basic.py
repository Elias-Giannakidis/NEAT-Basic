import random
import os
import neat

class Agent:
    def __init__(self):
        pass

    def function(self, x, y):
        return x + y

generation = 0
min_error = 0.1

def main(genomes, config):


    global generation
    generation += 1

    nets = []
    agents = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        agents.append(Agent())
        ge.append(genome)

    run = True
    end = False

    counter = 0
    while run and not end:

        counter += 1

        for i, agent in enumerate(agents):
            x = random.randint(0, 10)/20
            y = random.randint(0, 10)/20
            output = nets[i].activate((x, y))
            error = abs(output[0] - agent.function(x, y))
            print(error)
            ge[i].fitness += error**2

        for x, genome in enumerate(genomes):
            if genome[1].fitness > 200.0:
                agents.pop(x)
                nets.pop(x)
                ge.pop(x)
                genomes.pop(x)

        if len(agents) <= 0:
            end = True

def run(config_path):

    # Configuration the NEAT model.
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # configuration of population
    p = neat.Population(config)

    # Add the report operation. In the end of each episode it write a report.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # for population run the main function "n" times
    n = 100
    p.run(main, n)

if __name__ == "__main__":
    # Define the local direction.
    local_dir = os.path.dirname(__file__)
    # Make the path of configuration file
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    # Use the config_path to the run function.
    run(config_path)