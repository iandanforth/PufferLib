import math
import gymnasium as gym
import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.postprocess
import functools

class ContinuousCartPoleEnv(gym.Env):
    def __init__(self, render_mode='rgb_array'):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.x_threshold * 2, -np.finfo(np.float32).max, -self.theta_threshold_radians * 2, -np.finfo(np.float32).max]),
            high=np.array([self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold_radians * 2, np.finfo(np.float32).max]),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * float(action)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self, seed=None):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        if self.render_mode == 'rgb_array':
            return self.viewer.render(return_rgb_array=True)
        elif self.render_mode == 'human':
            return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def env_creator(name='continuous_cartpole'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array'):
    if name == 'continuous_cartpole':
        env_cls = ContinuousCartPoleEnv
    else:
        raise ValueError(f'Unknown environment: {name}')

    env = env_cls(render_mode=render_mode)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)
