from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, obs, cur_memory, ob, ob_last=None, local_map = None, imitation_learning=False):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        if imitation_learning:
            state = JointState(self.get_full_state(), ob, imitation_learning)
            action = self.policy.predict(state, obs)
        else:
            state = JointState(self.get_full_state(), ob, imitation_learning)
            state_last = JointState(self.get_full_state(), ob_last, imitation_learning)
            action = self.policy.predict(state_last, state, cur_memory, local_map)
        return action
