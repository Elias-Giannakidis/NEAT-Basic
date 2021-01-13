import neat
import random
import os

# This is the function and it related to the environment on another problem.
def function(x, y, z):
    output = (x + y + z)/100
    return output


def main(genomes, config):

    # Randomize the input that is the environment state of another problem.
    x = random.randint(0, 100)/101
    y = random.randint(0, 100)/101
    z = random.randint(0, 100)/101

    # Run the loop for each genome.
    for _, g in genomes:
        # restart the fitness function of genome.
        g.fitness = 0
        # Make the net according genome and configuration.
        net = neat.nn.FeedForwardNetwork.create(g, config)
        # Calculate the output according the current net.
        output = net.activate((x, y, z))
        # Calculate the error or gain function.
        error = function(x, y, z) - output[0]
        # Define the fitness of current genome.
        g.fitness += abs(error)


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
    n = 1000
    p.run(main, n)

if __name__ == "__main__":
    # Define the local direction.
    local_dir = os.path.dirname(__file__)
    # Make the path of configuration file
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    # Use the config_path to the run function.
    run(config_path)




