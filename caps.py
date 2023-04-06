import torch

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

import ray


from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch


from ray.rllib.utils.typing import TensorType



import scipy.stats as stats
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper

from typing import Type, Union, List
import functions as f

class CAPSTorchPolicy(PPOTorchPolicy):
    
    sigma = 0.01
    lambda_s = 1
    lambda_t = 1

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.previous_actions = torch.zeros(action_space.shape[0])
        self.lambda_t = config.get("lambda_t", 0.1)
        self.lambda_s = config.get("lambda_s", 0.1)
        self.sigma = config.get("sigma", 0.1)
    def loss(
    self,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
        # get the loss from the parent class
        loss = super().loss(model, dist_class, train_batch)
        
        # get the observations and actions
        obs, actions = train_batch["obs"], train_batch["actions"]
        
        # get the logits and the state of the model
        logits, _ = model({"obs": obs})
        
        # calculate the mean of L_T and L_S over the training batch
        L_S = 0


        #get a bunch of normal distribution around 
        dist = torch.distributions.Normal(obs, CAPSTorchPolicy.sigma )

        around_obs = dist.sample()

        logits_around, _ = model({"obs": around_obs})



        L_S = 0
        L_T = 0

        for i in range (len(train_batch["actions"])):


            # get the loss of the state around the observations
            L_S += torch.mean(abs(logits[i]-logits_around[i]))

            # get the loss of the actions around the observations
            if(i>0):
                L_T +=  f.action_dist(actions[i],actions[i-1])
            
        L_S = L_S / len(train_batch["actions"])
        L_T = L_T / len(train_batch["actions"])
        
        # add the loss of the state around the observations to the loss
        loss += CAPSTorchPolicy.lambda_s * L_S
        loss += CAPSTorchPolicy.lambda_t * L_T

        return loss


class PPOCAPSTrainer(PPOTrainer):
    def __init__(self, config=None, env=None):
        super().__init__(config=config, env=env)

    def get_default_policy_class(self, registry):

        return CAPSTorchPolicy

    def _init_optimizers(self):
        opt_type = self.config["optimizer"]["type"]
        if opt_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config["optimizer"]["lr"], eps=self.config["optimizer"]["epsilon"])
        else:
            return super()._init_optimizers()
