import neat
import random


def update(geneomes, config):
    for geneomeID, geneome in geneomes:
        net = neat.nn.FeedForwardNetwork.create(geneome, config)
        l = 100
        fit = l
        scalar = 100
        for i in range(l):
            input_ = random.randint(1,20)
            input_ = input_/(float(scalar))
            output = net.activate([input_])
            output[0] *= scalar
            #print "Input", input_ * scalar, "oputput", output[0]
            fit -= ((output[0] - ((scalar* input_) * 2)) **2)

        i = random.randint(1,20) /float(scalar)
        o = net.activate([i])[0] * scalar
        #print geneomeID, i, o
        geneome.fitness = fit




config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                 "config-multiplyer")

population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(5))

winner = population.run(update, 300)

net = neat.nn.FeedForwardNetwork.create(winner, config)

while True:
    val = input("Number: ")
    print 'values is', val
    if val == "q":
        break
    val = float(val)
    val /= 100
    print net.activate([val])[0]*100

