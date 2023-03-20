from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        #每个行人是需要一个目标来驱动行人运动的，但是机器人观测不到行人的full_state
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
