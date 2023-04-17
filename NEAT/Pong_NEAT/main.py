# https://neat-python.readthedocs.io/en/latest/xor_example.html
from pong import Game
import pygame
import neat
import os
import time
import pickle

width = 700
height = 500

window = pygame.display.set_mode((width, height))
game = Game(window, width, height)

class Pong:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball
        
    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
            
            keys = pygame.key.get_pressed()
            # Left paddle
            if keys[pygame.K_w]:
                game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                game.move_paddle(left=True, up=False)  
                
            # out = net.activate((self.right_paddle.y, self.ball.y, abs(self.ball.x - self.right_paddle.x)))
            # dec = out.index(max(out))
            # if dec == 0:
            #     pass
            # elif dec == 1:
            #     self.game.move_paddle(left=False, up=True)
            # else:
            #     self.game.move_paddle(left=False, up=False)
                
            self.game.loop()
            # print(game_info.left_score, game_info.right_score)
            #game.loop()
            self.game.draw(True, False)
            pygame.display.update()
        pygame.quit()
        
    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                    
            out1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.ball.x - self.left_paddle.x)))
            dec1 = out1.index(max(out1))
            if dec1 == 0:
                pass
            elif dec1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)
            
            out2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.ball.x - self.right_paddle.x)))
            dec2 = out2.index(max(out2))
            if dec2 == 0:
                pass
            elif dec2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)
            
            game_info = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()
            
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calc_fitness(genome1, genome2, game_info)
                break
                
    def calc_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits
        
        
    # Inputs: y-coordinate of paddle, y-coordinate of ball, absolute x distance of ball from paddle
    # Outputs: up, down, stay
def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    
    for i, (genome_id1, genome1) in enumerate(genomes):
        
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for j, (genome_id2, genome2) in enumerate(genomes[i+1:]):
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = Pong(window, width, height)
            game.train_ai(genome1, genome2, config)
            

def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-15')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    
    best = p.run(eval_genomes, 50)
    with open('best.pickle', 'wb') as f:
        pickle.dump(best, f)
        
def test_ai(config):
    with open('best.pickle', 'rb') as f:
        best = pickle.load(f)
    game = Pong(window, width, height)
    game.test_ai(best, config)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                         config_path)
    
    run_neat(config)
    test_ai(config)
    
    
    




