from caps import * 
import loader as l 
from cassie import CassieEnv
from ray.tune.registry import register_env
import torch 

torch.cuda.empty_cache()


register_env("cassie-v0", lambda config: CassieEnv(config))

loader = l.Loader(simdir="./sims/", logdir="/home/ajvendetta/ray_results")
config = loader.load_config("ConfigSimplified.yaml")
trainer = PPOCAPSConfig()
splitted = loader.split_config(config)
trainer = trainer.environment(**splitted.get("environment",{})).rollouts(**splitted.get("rollouts",{})).checkpointing(**splitted.get("checkpointing",{})).debugging(**splitted.get("debugging",{})).training(**splitted.get("training",{})).framework(**splitted.get("framework",{})).resources(**splitted.get("resources",{})).evaluation(**splitted.get("evaluation",{})).build()


# trainer = PPOCAPSTrainer(env='cassie-v0', config=config)
trainer.stop()