import os 
import logging as log 
import yaml 

class Loader():
    def __init__(self, logdir, simdir):
        self.logdir = logdir
        self.simdir = simdir


    def find_checkpoint(self,trainer_name="PPO"):

        checkpoint_path = None
        #load the trainer from the latest checkpoint if exists 
        #get the full directory of latest modified directory in the log_dir 
        if(os.path.exists(self.logdir)):
            latest_log_directory = max([d for d in os.listdir(self.logdir) if d.startswith(trainer_name+"_")], default=0)
            log.info("Found log directory with path " + os.path.join(self.logdir, str(latest_log_directory)) + " and latest log directory is (if 0 means no logs)" + str(latest_log_directory))
            #check that the folder is not empty
            if(latest_log_directory == 0):
                log.info("No checkpoint found in the log directory")
            else:     
                #get the latest directory in the latest log directory
                latest_directory = max([d.split("_")[-1] for d in os.listdir(os.path.join(self.logdir, latest_log_directory)) if d.startswith("checkpoint")], default=0)
                #load the trainer from the latest checkpoint
                checkpoint_path = os.path.join(self.logdir, latest_log_directory, "checkpoint_{}/".format(latest_directory, latest_directory))
                log.info("Found checkpoint in the log directory with path " + checkpoint_path) 
                return checkpoint_path
        return None
    #loads config from the yaml file and returns is as a dictionary
    def load_config(self, path):
        config = {
                    "framework": "torch",
                    "log_level": "WARN",
                    "num_gpus": 0,
                    "num_cpus": 8,
                    "num_workers": 8,
                    "num_envs_per_worker": 1,
                    "rollout_fragment_length": 100,
                    "train_batch_size": 10000,
                    "sgd_minibatch_size": 2000,
                    "observation_space":None,
                    "num_sgd_iter": 5,
                    "optimizer": {
                        "type": "Adam",
                        "lr": 3e-4,
                        "epsilon": 1e-5
                    },
                    "model": {
                        "conv_filters": None,
                        "fcnet_activation": "swish",
                        "fcnet_hiddens": [256,128,64,64],
                        "vf_share_layers": False,
                        "free_log_std": True,
                    },
                    "entropy_coeff": 0.01,
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "kl_coeff": 0.5,
                    "clip_param": 0.2,


                    "batch_mode": "complete_episodes",
                    "observation_filter": "NoFilter",
                    "reuse_actors": True,
                    "disable_env_checking": True,
                    "num_gpus_per_worker": 0,
                  # Evaluation parameters
                    "evaluation_interval": 2,
                    "evaluation_num_episodes": 10,
                    "evaluation_config": {
                        "env": "cassie-v0",
                        "seed": 1234,

                    }

                }
        return config 
        # with open(path, 'r') as stream:
        #     try:
        #         con = yaml.safe_load(stream)
        #         print(con)
        #         return yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)

    def recover_weights(self,Trainer,checkpoint_path, config):
        #check that checkpoint path is valid and that it does not end with checkpoint_0
        if(checkpoint_path is not None and checkpoint_path.split("/")[-2].split("_")[-1] != "0"):
            #load the a temporary trainer from the checkpoint
            temp = Trainer(config, "cassie-v0")
            temp.restore(checkpoint_path)


            # Get policy weights
            policy_weights = temp.get_policy().get_weights()
            # Destroy temp
            temp.stop()
            log.info("Weights loaded from checkpoint successfully")
            return policy_weights
        else: 
            temp = None
            log.error("Weights are not valid for this trainer")
            return None