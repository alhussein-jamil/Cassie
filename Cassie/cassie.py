import constants as c 
import gymnasium.utils as utils 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import functions as f 
import numpy as np
import gymnasium as gym 
from gymnasium.spaces import Box
import mujoco as m 
import torch
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.agents.ppo import PPOTrainer
from loader import Loader
import mediapy as media
import os

from ray.tune.registry import register_env
import argparse
from caps import *
import logging as log 
import shutil
from ray.rllib.algorithms.registry import POLICIES
import numpy as np 
log.basicConfig(level=log.DEBUG)
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict, Tuple
import os
import json
from ray.tune.logger import TBXLogger, UnifiedLogger
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import UnifiedLogger
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray import tune


class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }


    def __init__(self,config,  **kwargs):
        utils.EzPickle.__init__(self, config, **kwargs)
        self._terminate_when_unhealthy = config.get("terminate_when_unhealthy", True)
        self._healthy_z_range = config.get("healthy_z_range", (0.35, 2.0))

        low, high = [], []
        for key in c.actuator_ranges.keys():
            low.append(c.actuator_ranges[key][0])
            high.append(c.actuator_ranges[key][1])
        self.action_space = gym.spaces.Box(np.float32(np.array(low)), np.float32(np.array(high)))

        self._reset_noise_scale = config.get("reset_noise_scale", 1e-2)
        self.phi, self.steps, self.gamma_modified = 0, 0, 1
        self.previous_action = torch.zeros(10)
        self.gamma = config.get("gamma", 0.99)
        self.rewards = {"R_biped": 0, "R_cmd": 0, "R_smooth": 0}

        low, high = [-3] * 23 + [-1, -1], [3] * 23 + [1, 1]
        self.observation_space = Box(low=np.float32(np.array(low)), high=np.float32(np.array(high)), shape=(25,))

        MujocoEnv.__init__(
            self,
            config.get("model_path", "C:\\Users\\Ajvendetta\\Downloads\\cassiePallas-main\\cassiePallas-main\\cassie.xml"),
            20,
            render_mode=config.get("render_mode", None),
            observation_space=self.observation_space,
            **kwargs
        )

    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        #it is healthy if in range and one of the feet is on the ground
        is_healthy = min_z < self.data.qpos[2] < max_z 
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if (self._terminate_when_unhealthy or self.steps>c.MAX_STEPS)  else False
        return terminated
    
    def _get_obs(self):
        p =np.array ([np.sin((2*np.pi*(self.phi))),np.cos((2*np.pi*(self.phi)))])
        temp = []
        #normalize the sensor data using sensor_ranges self.data.sensor('pelvis-orientation').data
        for key in c.sensor_ranges.keys():
            temp.append(f.normalize(key,self.data.sensor(key).data))
        temp = np.array(np.concatenate(temp))

        #getting the read positions of the sensors and concatenate the lists
        return np.concatenate([temp,p])

    
    #computes the reward
    def compute_reward(self,action):

        # Extract some proxies
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        
        qpos = qpos[pos_index]
        qvel=qvel[vel_index]

        #Feet Contact Forces 
        contact_force_right_foot = np.zeros(6)
        m.mj_contactForce(self.model,self.data,0,contact_force_right_foot)
        contact_force_left_foot = np.zeros(6)
        m.mj_contactForce(self.model,self.data,1,contact_force_left_foot)


        #Some metrics to be used in the reward function
        q_vx = 1-np.exp(c.multiplicators["q_vx"]*np.linalg.norm(np.array([qvel[0]]) - np.array([c.X_VEL]))**2)
        q_vy = 1-np.exp(c.multiplicators["q_vy"]*np.linalg.norm(np.array([qvel[1]]) - np.array([c.Y_VEL]))**2)
        q_vz = 1-np.exp(c.multiplicators["q_vz"]*np.linalg.norm(np.array([qvel[2]]) - np.array([c.Z_VEL]))**2)

        q_left_frc = 1.0 - np.exp(c.multiplicators["q_frc"] * np.linalg.norm(contact_force_left_foot)**2)
        q_right_frc = 1.0 - np.exp(c.multiplicators["q_frc"] * np.linalg.norm(contact_force_right_foot)**2)
        q_left_spd = 1.0 - np.exp(c.multiplicators["q_spd"] * np.linalg.norm(qvel[12])**2)
        q_right_spd = 1.0 - np.exp(c.multiplicators["q_spd"] * np.linalg.norm(qvel[19])**2)
        q_action_diff = 1 - np.exp(c.multiplicators["q_action"] * float(f.action_dist(torch.tensor(action).reshape(1,-1),torch.tensor(self.previous_action).reshape(1,-1))))
        q_orientation = 1 -np.exp(c.multiplicators["q_orientation"]*(1-((self.data.sensor('pelvis-orientation').data.T)@(c.FORWARD_QUARTERNIONS))**2))
        q_torque = 1 - np.exp(c.multiplicators["q_torque"]*np.linalg.norm(action))
        q_pelvis_acc = 1 - np.exp(c.multiplicators["q_pelvis_acc"] * (np.linalg.norm(self.data.sensor('pelvis-angular-velocity').data) ))#+ np.linalg.norm(self.data.sensor('pelvis-linear-acceleration').data-self.model.opt.gravity.data)))

        self.exponents = {'q_vx':np.linalg.norm(np.array([qvel[0]]) - np.array([c.X_VEL]))**2, 'q_vy' : np.linalg.norm(np.array([qvel[1]]) - np.array([c.Y_VEL]))**2, 'q_vz' : np.linalg.norm(np.array([qvel[2]]) - np.array([c.Z_VEL]))**2, 'q_left_frc' : np.linalg.norm(contact_force_left_foot)**2, 'q_right_frc' : np.linalg.norm(contact_force_right_foot)**2, 'q_left_spd' : np.linalg.norm(qvel[12])**2, 'q_right_spd' : np.linalg.norm(qvel[19])**2, 'q_action_diff' : float(f.action_dist(torch.tensor(action).reshape(1,-1),torch.tensor(self.previous_action).reshape(1,-1))), 'q_orientation' : (1-((self.data.sensor('pelvis-orientation').data.T)@(c.FORWARD_QUARTERNIONS))**2), 'q_torque' : np.linalg.norm(action), 'q_pelvis_acc' : np.linalg.norm(self.data.sensor('pelvis-angular-velocity').data) + np.linalg.norm(self.data.sensor('pelvis-linear-acceleration').data-self.model.opt.gravity.data)}
        self.used_quantities = {'q_vx': q_vx, 'q_vy' : q_vy, 'q_vz' : q_vz, 'q_left_frc' : q_left_frc, 'q_right_frc' : q_right_frc, 'q_left_spd' : q_left_spd, 'q_right_spd' : q_right_spd, 'q_action_diff' : q_action_diff, 'q_orientation' : q_orientation, 'q_torque' : q_torque, 'q_pelvis_acc' : q_pelvis_acc}
        #Responsable for the swing and stance phase
        I = lambda phi,a,b : f.p_between_von_mises(a,b,c.KAPPA,phi)

        I_swing_frc = lambda phi : I(phi,c.a_swing,c.b_swing)
        I_swing_spd = lambda phi : I(phi, c.a_swing,c.b_swing)
        I_stance_spd = lambda phi : I(phi, c.a_stance,c.b_stance)
        I_stance_frc = lambda phi : I(phi, c.a_stance,c.b_stance)

        C_frc = lambda phi : c.c_swing_frc * I_swing_frc(phi) + c.c_stance_frc * I_stance_frc(phi) 
        C_spd = lambda phi :  c.c_swing_spd * I_swing_spd(phi) + c.c_stance_spd * I_stance_spd(phi)
        

        R_cmd = - 1.0*q_vx - 1.0*q_vy - 1.0*q_orientation - 0.5*q_vz
        R_smooth = -1.0*q_action_diff - 1.0* q_torque - 1.0*q_pelvis_acc
        R_biped = 0
        R_biped += C_frc(self.phi+c.THETA_LEFT) * q_left_frc
        R_biped += C_frc(self.phi+c.THETA_RIGHT) * q_right_frc
        R_biped += C_spd(self.phi+c.THETA_LEFT) * q_left_spd
        R_biped += C_spd(self.phi+c.THETA_RIGHT) * q_right_spd

        reward = 2.5 + 0.5 * R_biped  +  0.375* R_cmd +  0.125* R_smooth
        
        self.rewards = {'R_biped': R_biped, 'R_cmd': R_cmd, 'R_smooth': R_smooth}

        return reward
    
    #step in time
    def step(self, action):
        #clip the action to the ranges in action_space (done inside the config that's why removed)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        
        reward = self.compute_reward(action)

        terminated = self.terminated

        self.steps +=1 
        self.phi+= 1.0/c.STEPS_IN_CYCLE
        self.phi = self.phi % 1 

        self.previous_action = action 

        self.gamma_modified *= self.gamma
        info = {}
        info['custom_rewards'] = self.rewards
        info['custom_quantities'] = self.used_quantities
        info['custom_metrics'] = {'distance' : self.data.qpos[0], 'height' : self.data.qpos[2]}
        return observation, reward, terminated, False, info

    #resets the simulation
    def reset_model(self):
        m.mj_inverse(self.model, self.data)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.previous_action = np.zeros (10)
        self.phi = 0 
        self.steps = 0 
        self.rewards = {"R_biped":0,"R_cmd":0,"R_smooth":0}

        self.gamma_modified = 1
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    

class MyCallbacks(DefaultCallbacks):


    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        for key in episode._last_infos['agent0'].keys():
            for key2 in episode._last_infos['agent0'][key].keys():
                if(key+"_"+key2 not in episode.user_data.keys()):
                    episode.user_data[key+"_"+key2] = []
                    episode.hist_data[key+"_"+key2] = []
                episode.user_data[key+"_"+key2].append(episode._last_infos['agent0'][key][key2])
                episode.hist_data[key+"_"+key2].append(episode._last_infos['agent0'][key][key2])
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        for key in episode._last_infos['agent0'].keys():
            for key2 in episode._last_infos['agent0'][key].keys():
                episode.custom_metrics[key+"_"+key2] = (np.mean(episode.user_data[key+"_"+key2]))
                episode.hist_data[key+"_"+key2] = episode.user_data[key+"_"+key2]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True


    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
    
