import gym
from gym import spaces

from quad_multi_sim.quadrotor_env import QuadrotorEnv


class MultiQuadPolicy:
    def __init__(self, policy_cls, num_quads):
        self.policies = []
        for i in range(num_quads):
            policy = policy_cls()
            self.policies.append(policy)

    def reset(self):
        for p in self.policies:
            p.reset()

    def step(self, obs):
        actions = []
        for o, p in zip(obs, self.policies):
            action = p.step(o)
            actions.append(action)

        return actions


class MultiQuadEnv(gym.Env):
    def __init__(
            self,
            num_quads, quad, raw_control, raw_control_zero_middle,
            dyn_randomize_every=None, dyn_randomization_ratio=None,
            sense_noise=None, init_random_state=False, obs_repr="xyz_vxyz_rot_omega",
    ):
        self.num_quads = num_quads

        self.envs = []
        for i in range(num_quads):
            env = QuadrotorEnv(
                dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
                dynamics_randomize_every=dyn_randomize_every, dynamics_randomization_ratio=dyn_randomization_ratio,
                sense_noise=sense_noise, init_random_state=init_random_state, obs_repr=obs_repr
            )

            self.envs.append(env)

        self.action_space = spaces.Tuple([e.action_space for e in self.envs])
        self.observation_space = spaces.Tuple([e.observation_space for e in self.envs])

        # TODO: temporary fixes
        # TODO: get rid of this!!! We should use standard gym.Env interface
        self.control_freq = self.envs[0].control_freq

    def reset(self):
        states = [e.reset() for e in self.envs]
        return states

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []

        for action, e in zip(actions, self.envs):
            o, rew, done, info = e.step(action)
            obs.append(o)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        return obs, rewards, dones, infos

    def render(self, mode='human', **kwargs):
        res = []
        for e in self.envs:
            res.append(e.render(mode=mode, **kwargs))

        return res

