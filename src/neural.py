from __future__ import print_function
import retro
import numpy as np
import cv2
import neat
import pickle
import glob, os
import visualize
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/bin/'



imgarray = []

def eval_genomes(genomes,config):

    for genome_id, genome in genomes:

        ob = env.reset()
        
        inx, iny, inc = env.observation_space.shape
        
        inx = int(inx/8)
        iny = int(iny/8)
        print(inx, iny)
 

        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)

        current_max_fitness = 0
        fitness_current = 0
        counter = 0
        xpos = 0
        xpos_max = 0
    
        done = False

        while not done:

            env.render()

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            

            for x in ob:
                for y in x:
                    imgarray.append(y)  #NEAT needs 1dim array
            
            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)

            imgarray.clear()

            xpos = info['x']
            # xpos_end = info['screen_x_end']
            # print(xpos)
            # print(xpos_end)

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos
            
            if xpos == 9767:
                fitness_current += 100000
                done = True
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            
            if done or counter == 1400:
                done = True
                print(genome_id,fitness_current)
            
            genome.fitness = fitness_current


def load_last_checkpoint():
    try:
        os.chdir('../checkpoints/model1')
        checkpoints = [f for f in glob.glob('neat-checkpoint-*')]
        checkpoints = [int(f[16:])for f in checkpoints]
        checkpoints.sort()
        return neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(checkpoints[-1]))
    except:
        print('No checkpoints in our folder, starting training from generation 0')
        return neat.Population(config)

if __name__ == '__main__':

    #Here we create our game enviroment and load our network config

    env = retro.make('SonicTheHedgehog-Genesis' , 'GreenHillZone.Act1', record='../records')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward')

    # Restore the last checkpoint if exist, else starts from zero:
    # p = load_last_checkpoint()

    # Uncomment to restore a selected checkpoint if don't want to restore last checkpoint 
    p = neat.Checkpointer.restore_checkpoint('../checkpoints/model1/neat-checkpoint-97') #try 1,22,97


    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    print(stats)
    p.add_reporter(stats)

    # Every 5 generation save a checkpoint
    p.add_reporter(neat.Checkpointer(1,filename_prefix='../checkpoints/model1/neat-checkpoint-'))

    winner = p.run(eval_genomes)

 
    # uncoment to draw the network
    # visualize.draw_net(config, winner, True)


    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)



