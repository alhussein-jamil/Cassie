from ray.rllib.agents.ppo import PPOTrainer
from loader import Loader
import cv2 
import os
from cassie import CassieEnv
from ray.tune.registry import register_env
import argparse
from caps import *
import logging as log 
log.basicConfig(level=log.DEBUG)

if __name__ == "__main__":
    #To call the function I wan to use the following command: python run.py -clean --simdir="" --logdir=""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-cleanrun", action="store_true", help="Runs without loading the previous simulation")
    argparser.add_argument("-simdir", "--simdir", type=str, help="Simulation directory")
    argparser.add_argument("-logdir", "--logdir", type=str, help="Log directory")
    argparser.add_argument("-caps", action="store_true", help="Uses CAPS regularization")
    argparser.add_argument("--config", type=str, help="Path to config file")
    argparser.add_argument("--simfreq", type=int, help="Simulation frequency")
    argparser.add_argument("--checkfreq", type=int, help="Checkpoint frequency")
    args = argparser.parse_args()

    if(args.simfreq is not None):
        simulation_frequency = args.simfreq
        log.info("Simulation frequency: {}".format(simulation_frequency))
    else:
        sim_freq = 10

    if(args.checkfreq is not None):
        checkpoint_frequency = args.checkfreq
        log.info("Checkpoint frequency: {}".format(simulation_frequency))
    else:
        check_freq = 5

    clean_run = args.cleanrun
    if(clean_run):
        log.info("Running clean run")
    else:
        log.info("Resuming from previous run")
    if(args.simdir is not None):
        sim_dir = args.simdir
        log.info("Simulation directory: {}".format(sim_dir))
    else:
        sim_dir = "./sims/"
    
    if(args.logdir is not None):
        log_dir = args.logdir
        log.info("Log directory: {}".format(log_dir))
    else:
        log_dir = "/home/alhussein.jamil/ray_results"



    register_env("cassie-v0", lambda config: CassieEnv(config))

    loader = Loader(logdir = log_dir, simdir = sim_dir)
    config = "config.yaml"
    
    if(args.config is not None):
        config = args.config


    config = loader.load_config(config)

    if(not args.caps):
        log.info('Running without CAPS regularization')
        Trainer = PPOTrainer
    else:
        log.info('Running with CAPS regularization')
        Trainer = PPOCAPSTrainer
        from ray.rllib.algorithms.registry import POLICIES
        # register the policy
        POLICIES["CAPSTorchPolicy"] = CAPSTorchPolicy

    if(not clean_run):
        checkpoint_path = loader.find_checkpoint(Trainer.__name__)
        weights = loader.recover_weights(Trainer, checkpoint_path, config)

    
    trainer = Trainer(config=config, env="cassie-v0")
    
    if(not clean_run and weights is not None):
        if(checkpoint_path is not None and weights.keys() == trainer.get_policy().get_weights().keys()) :
    
            trainer.get_policy().set_weights(weights)
            print("Weights loaded successfully")


    # Define video codec and framerate
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30

    # Training loop
    max_test_i = 0
    checkpoint_frequency = 5
    simulation_frequency = 10
    env = CassieEnv({})
    env.render_mode = "rgb_array"

    # Create sim directory if it doesn't exist
    if(not os.path.exists(sim_dir)):
        os.makedirs(sim_dir)

    # Find the latest directory named test_i in the sim directory
    latest_directory = max([int(d.split("_")[-1]) for d in os.listdir(sim_dir) if d.startswith("test_")], default=0)
    max_test_i = latest_directory + 1

    # Create folder for test
    test_dir = os.path.join(sim_dir, "test_{}".format(max_test_i))
    os.makedirs(test_dir, exist_ok=True)

    # Define video codec and framerate
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30

    # Set initial iteration count
    i = trainer.iteration if hasattr(trainer, "iteration") else 0

    while True:
            # Train for one iteration
            result = trainer.train()
            i += 1
            print("Episode Reward Mean for iteration {} is {}".format(i, result["episode_reward_mean"]))

            # Save model every 10 epochs
            if i % checkpoint_frequency == 0:
                checkpoint_path = trainer.save()
                print("Checkpoint saved at", checkpoint_path)

            # Run a test every 20 epochs
            if i % simulation_frequency == 0:
                
                #make a steps counter
                steps = 0

                # Run test
                video_path = os.path.join(test_dir, "sim_{}.mp4".format(i))

                env.reset()
                obs = env.reset()[0]
                done = False
                frames = []

                while not done:

                    # Increment steps
                    steps += 1
                    action = trainer.compute_single_action(obs)
                    obs, _, done, _, _ = env.step(action)
                    frame = env.render()
                    frames.append(frame)

                # Save frames as video
                height, width, _ = frames[0].shape
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                for frame in frames:
                    video_writer.write(frame)
                video_writer.release()
                print("Test saved at", video_path)
                # Increment test index
                max_test_i += 1