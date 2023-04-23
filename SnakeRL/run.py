from snake_env import SnakeEnv
from apple import Apple
from snake import Snake

import pygame
import random
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
import matplotlib.pyplot as plt

ray.init()
print("Ray initialized")


def env_creator(env_config):
    return SnakeEnv(env_config)


register_env("snake-v0", env_creator)

# read the yaml as dictionary
import yaml

with open("SnakeConfig.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
        if config is None:
            raise Exception("Invalid YAML file")
    except yaml.YAMLError as exc:
        print(exc)

print("Config loaded")
fps = 10
# snakie = SnakeEnv({'render_mode':"human"})
# snakie.reset()
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True
#             pygame.quit()
#     snakie.render("human")
#     #wait for 1/fps seconds
#     pygame.time.wait(1000//fps)

#     action = snakie.action_space.sample()
#     obs, reward, done, _ , _= snakie.step(action)
#     if done:
#         snakie.reset()


trainer = PPOTrainer(config=config, env="snake-v0")

snakie = SnakeEnv({})
for i in range(100):
    result = trainer.train()
    print(
        "iteration: ",
        i,
        "episode_reward_mean: ",
        result["episode_reward_mean"],
        "episode_reward_min: ",
        result["episode_reward_min"],
        "episode_reward_max: ",
        result["episode_reward_max"],
    )

    if i % 10 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

    if i % 10 == 0:
        snakie.reset()
        done = False
        frames = []
        while not done:
            action = trainer.compute_single_action(snakie._get_obs())
            obs, reward, done, _, _ = snakie.step(action)
            frames.append(snakie.render())

            # wait for 1/fps seconds
            # pygame.time.wait(1000//fps)
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         done = True
            #         pygame.quit()

        # save the video
        import imageio

        imageio.mimsave("snake" + str(i) + ".gif", frames, fps=fps)
        # pygame.quit()
