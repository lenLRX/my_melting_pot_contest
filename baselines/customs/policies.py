from typing import Tuple, List, Optional, Union, Dict

import dm_env
import numpy as np
import torch

from meltingpot.utils.policies import policy
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.policy import sample_batch
from ray.rllib.policy.sample_batch import SampleBatch

from gymnasium.spaces import Box

from ray.rllib.utils.typing import (
    AgentID,
    AlgorithmConfigDict,
    ModelGradients,
    ModelWeights,
    PolicyID,
    PolicyState,
    T,
    TensorStructType,
    TensorType,
)

_IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']


SPECIAL_COLOR = [(115, 115, 115), (20, 41, 81), (129, 34, 53), (252, 252, 106), (34, 129, 109), (44, 60, 91)]

class MyCustomPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.ac = 0
        self.pixel_scale = 1 / 255

        super().__init__(observation_space, action_space, config)
        print("observation_space:")
        print(observation_space)
        print("action_space:")
        print(action_space)
        #print("_enable_rl_module_api:", self.config.get("_enable_rl_module_api", False))
        #print("self.model", str(self.model))


    @override(TorchPolicyV2)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        print("MyCustomPolicy compute_actions")
        assert False
        return super().compute_actions(obs_batch,
                        state_batches,
                        prev_action_batch,
                        prev_reward_batch,
                        info_batch,
                        episodes,
                        explore,
                        timestep,
                        **kwargs)

    #def extra_grad_process(self, local_optimizer, loss):
    #    assert False
    #    return None


    def compute_actions_from_input_dict(
            self,
            input_dict: Union[SampleBatch, Dict[str, TensorStructType]],
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            episodes: Optional[List["Episode"]] = None,
            **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        #print(input_dict)

        raw_img = input_dict["obs"]["RGB"]
        batch_size = raw_img.shape[0]

        extra_img = np.zeros((batch_size, 11, 11, 6), dtype=np.uint8)

        for bi in range(batch_size):
            for h in range(11):
                for w in range(11):
                    l = tuple(raw_img[bi, h, w].tolist())
                    for c in range(6):
                        if l == SPECIAL_COLOR[c]:
                            extra_img[bi, h, w, c] = 1

        #input_dict["obs"]["RGB"] = input_dict["obs"]["RGB"]*self.pixel_scale
        input_dict["obs"]["extra_viz"] = extra_img
        #print("input rgb type: ", type(input_dict["obs"]["RGB"]))

        #print(input_dict["obs"])
        #print(input_dict["prev_actions"])
        #print("state_in_0 shape", input_dict["state_in_0"].shape)
        #print(input_dict["state_in_1"])
        #print("state_in_1 shape", input_dict["state_in_1"].shape)

        #assert self.ac < 10
        #self.ac += 1
        return super().compute_actions_from_input_dict(input_dict, explore, timestep)

    def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
    ):
        #print(sample_batch['obs'].keys())
        batch_reward = sample_batch[SampleBatch.REWARDS]
        #print("reward shape: {}".format(batch_reward.shape))
        #print(f"max reward {batch_reward.max()}")

        raw_img = sample_batch["obs"]["RGB"]
        raw_img_new = sample_batch["new_obs"]["RGB"]
        batch_size = raw_img.shape[0]

        #print("raw_img shape", raw_img.shape)
        #print(raw_img[0])

        extra_img = np.zeros((batch_size, 11, 11, 6), dtype=np.uint8)
        extra_img_new = np.zeros((batch_size, 11, 11, 6), dtype=np.uint8)

        for bi in range(batch_size):
            for h in range(11):
                for w in range(11):
                    l = tuple(raw_img[bi, h, w].tolist())
                    nl = tuple(raw_img_new[bi, h, w].tolist())
                    for c in range(6):
                        r = SPECIAL_COLOR[c]
                        #print(l, r)
                        if l == r:
                            extra_img[bi, h, w, c] = 1
                            #print(f"{bi}, {h}, {w}, {c}")
                        if nl == r:
                            extra_img_new[bi, h, w, c] = 1
                            #print(f"{bi}, {h}, {w}, {c}")

        #sample_batch["obs"]["RGB"] = sample_batch["obs"]["RGB"] * self.pixel_scale
        sample_batch["obs"]["extra_viz"] = extra_img
        sample_batch["new_obs"]["extra_viz"] = extra_img_new
        inventory_sum = sample_batch["obs"]["INVENTORY"].sum(-1)
        next_inventory_sum = sample_batch["new_obs"]["INVENTORY"].sum(-1)
        inventory_diff = next_inventory_sum - inventory_sum
        batch_reward = batch_reward + np.maximum(inventory_diff, 0)*0.01
        sample_batch[SampleBatch.REWARDS] = batch_reward
        #print(f"inventory_sum shape {inventory_sum.shape}")

        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

class EvalPolicy(policy.Policy):
  """ Loads the policies from  Policy checkpoints and removes unrequired observations
  that policies cannot expect to have access to during evaluation.
  """
  def __init__(self,
               chkpt_dir: str,
               policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
    
    policy_path = f'{chkpt_dir}/{policy_id}'
    self._policy = Policy.from_checkpoint(policy_path)
    self._prev_action = 0
  
  def initial_state(self) -> policy.State:
    """See base class."""

    self._prev_action = 0
    state = self._policy.get_initial_state()
    self.prev_state = state
    return state

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""

    observations = {
        key: value
        for key, value in timestep.observation.items()
        if key not in _IGNORE_KEYS
    }

    # We want the logic to be stateless so don't use prev_state from input
    action, state, _ = self._policy.compute_single_action(
        observations,
        self.prev_state,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    self.prev_state = state
    return action, state

  def close(self) -> None:

    """See base class."""