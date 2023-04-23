from snake import Snake
from apple import Apple
import pygame
import random 
import gymnasium as gym
import numpy as np

import gymnasium.utils as utils 

from gymnasium.spaces import Discrete, Box


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self,config,  **kwargs):
        utils.EzPickle.__init__(self, config, **kwargs)

        self.screen_width = config.get('screen_width', 640)
        self.screen_height = config.get('screen_height', 480)
        self.block_size =  config.get('block_size', 20)
        self.snake = Snake(self.screen_width, self.screen_height, self.block_size)
        self.apple = self.generate_apple()
        if(config.get('render_mode',"rgb_array") == "human"):
            pygame.init()
            self.screen =  pygame.display.set_mode((self.screen_width, self.screen_height))
        #self.observation_space = Box(low=0, high=1, shape=(self.screen_height//self.block_size*self.screen_width//self.block_size* 3,), dtype=np.uint8)
        self.observation_space = Box(low = np.array([-self.screen_width//self.block_size,- self.screen_height//self.block_size, -1,-1]), high = np.array( [self.screen_width//self.block_size, self.screen_height//self.block_size, 1,1]), shape = (4,), dtype = np.float32)
        self.action_space = Box(low=0, high=3, shape=(1,), dtype=np.uint8)

        self.reset()

    def compute_reward(self, action):
        self.reward = 0
        #if eaten apple add 10 to the reward
        if self.snake.head == self.apple.position:
            self.reward += 1
        #if snake collides with wall or body subtract 10 from the reward
        if self.snake.head[0] < 0 or self.snake.head[0] >= self.screen_width or self.snake.head[1] < 0 or self.snake.head[1] >= self.screen_height:
            self.reward -= 1

        if self.snake.head in self.snake.body[1:]:
            self.reward -= 1
        #add some reward proportional to the distance from the apple
        self.reward += np.exp(-3*np.sqrt(((self.snake.head[0] - self.apple.position[0])/self.block_size)**2 + ((self.snake.head[1] - self.apple.position[1])/self.block_size)**2)/((self.screen_width/self.block_size+self.screen_height/self.block_size)/2.0))
        #print(np.exp(-np.sqrt(((self.snake.head[0] - self.apple.position[0])/self.block_size)**2 + ((self.snake.head[1] - self.apple.position[1])/self.block_size)**2)/((self.screen_width/self.block_size+self.screen_height/self.block_size)/2.0)))
        return self.reward


    def reset(self, iteration=0, seed=None, options=None):
        # This function resets the game state and returns the initial observation
        # of the game state.

        # Initialize the snake and apple
        self.snake = Snake(self.screen_width, self.screen_height, self.block_size)
        self.snake.head = (self.screen_width // 2, self.screen_height // 2)
        self.snake.body = [(self.screen_width // 2, self.screen_height // 2)]
        self.snake.direction = (1, 0)
        self.snake.grow = False
        self.apple = self.generate_apple()
        self.score = 0
        self.done = False
        self.reward = 0
        # Return the initial observation of the game state
        return self._get_obs(), {}


    
    def step(self, action):
        # Change snake direction
        self.snake.change_direction(action)
        # Move snake
        self.snake.move()
        self.compute_reward(action)
        # Check if snake eats apple
        if self.snake.head == self.apple.position:
            self.score += 1
            self.snake.grow = True
            self.apple = self.generate_apple()

        # Check if snake collides with wall
        if self.snake.head[0] < 0 or self.snake.head[0] >= self.screen_width or self.snake.head[1] < 0 or self.snake.head[1] >= self.screen_height:
            self.done = True

        # Check if snake collides with body
        if self.snake.head in self.snake.body[1:]:
            self.done = True

        return self._get_obs(),self.reward, self.done, False, {}
    
    # Make a random apple
    def generate_apple(self):
        # Make a random x and y location
        x = random.randint(0, (self.screen_width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.screen_height - self.block_size) // self.block_size) * self.block_size
        # Make an apple with those x and y values

        # Check if the apple is in the snake's body
        # If it is, generate a new apple
        while (x,y) in self.snake.body or (x,y) == self.snake.head:
            x = random.randint(0, (self.screen_width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.screen_height - self.block_size) // self.block_size) * self.block_size
        return Apple(x, y, self.block_size)
    
    def render(self, mode = "rgb_array"):
        if(mode == "rgb_array"):
            image = np.zeros((self.screen_height, self.screen_width, 3),dtype=np.uint8)
            #make the image white 
            image[:,:,:] = [255, 255, 255]
            #red for the apple 
            image[self.apple.position[1]:self.apple.position[1]+self.block_size, self.apple.position[0]:self.apple.position[0]+self.block_size, :] = [255, 0, 0]

            #green for the snake
            for pos in self.snake.body:
                image[pos[1]:pos[1]+self.block_size, pos[0]: pos[0]+self.block_size, :] = [0, 255, 0]
            #blue for the head
            image[self.snake.head[1]: self.snake.head[1]+self.block_size, self.snake.head[0]:self.snake.head[0]+self.block_size, :] = [0, 0, 255]
            return image
        else:    
            # Fill the screen with white background
            self.screen.fill((255, 255, 255))
            # Draw the snake on the screen
            self.snake.draw(self.screen)
            # Draw the apple on the screen
            self.apple.draw(self.screen)
            # Update the screen to show the changes
            pygame.display.update()
            # Wait 100 milliseconds
            pygame.time.delay(100)


    def in_grid_bounds(self, pos):
        return 0 <= pos[0] < self.screen_width and 0 <= pos[1] < self.screen_height
    def _get_obs(self):
        obs = np.zeros(4)
        obs[0] = (self.snake.head[0] - self.apple.position[0])/self.block_size
        obs[1] = (self.snake.head[1] - self.apple.position[1])/self.block_size
        obs[2] = self.snake.direction[0]
        obs[3] = self.snake.direction[1]
        return obs 
        # obs = np.zeros((self.screen_height//self.block_size, self.screen_width//self.block_size, 3), dtype=np.uint8)
        # obs[self.apple.position[1]//self.block_size, self.apple.position[0]//self.block_size, 0] = 1
        
        # for pos in self.snake.body:
        #     obs[pos[1]//self.block_size, pos[0]//self.block_size, 1] = 1
        # if(self.in_grid_bounds(self.snake.head)):
        #     obs[self.snake.head[1]//self.block_size, self.snake.head[0]//self.block_size, 2] = 1
        return obs.flatten()
    
