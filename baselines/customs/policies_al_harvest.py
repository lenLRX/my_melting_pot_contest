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

class MyCustomHarvestPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.ac = 0
        self.pixel_scale = 1 / 255

        print("observation_space:")
        print(observation_space)
        print("action_space:")
        print(action_space)

        super().__init__(observation_space, action_space, config)

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
        return super().compute_actions_from_input_dict(input_dict, explore, timestep)

    def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
    ):
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