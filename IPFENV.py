import numpy as np
import pygame
from gym.utils import EzPickle
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
import IPF_Env


class raw_env(IPF_Env.SimpleEnvIPF):
    def __init__(self, max_cycles=25, continuous_actions=False):
        scenario = Scenario()
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata["name"] = "simple_ipf"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

MAX_POTENTIAL = 5
MIN_POTENTIAL = -3

kernel_size = 13
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)


def numpy_conv(inputs, filter):
    H, W = inputs.shape
    filter_size = filter.shape[0]
    # filter_center = int(filter_size / 2.0)
    # filter_center_ceil = int(np.ceil(filter_size / 2.0))

    input_new = np.zeros((H + filter_size - 1, W + filter_size - 1))
    result = np.zeros((H + filter_size - 1, W + filter_size - 1))
    start = int(filter_size / 2)
    isodd = filter_size % 2 == 1
    if not isodd:
        print("error kernel not odd")
        return
    endH = start + H
    endW = start + W

    input_new[start:endH, start:endW] = inputs

    for r in range(start, endH):
        for c in range(start, endW):
            # 池化大小的输入区域
            cur_input = input_new[r - start:r + start + 1, c - start:c + start + 1]
            # 和核进行乘法计算
            cur_output = cur_input * filter
            # 再把所有值求和
            conv_sum = np.sum(cur_output)
            # 当前点输出值
            result[r, c] = conv_sum

    return result[start:endH, start:endW]


def clamp(min, max, value):
    if value < min: return min
    if value > max: return max
    return value


class Scenario(BaseScenario):

    def __init__(self):
        self.cam_range = None

    def make_world(self):
        self.iter_count = 8
        self.width = 70
        self.height = 70
        self.ipfmap = np.zeros((self.iter_count, self.width, self.height), dtype=float)

        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.ipfmap = np.zeros((self.iter_count, self.width, self.height), dtype=float)
        self.updateipf(world)

    def pos2ipfmap(self, x, y):
        # pos in range[-1 , 1] to range [0, 1] first
        cam_range = self.cam_range
        y *= (
            -1
        )  # this makes the display mimic the old pyglet setup (ie. flips image)
        x = (
                (x / cam_range) * self.width // 2 * 0.9
        )  # the .9 is just to keep entities from appearing "too" out-of-bounds
        y = (y / cam_range) * self.height // 2 * 0.9
        x += self.width // 2
        y += self.height // 2

        return clamp(0, self.width - 1, int(x)), clamp(0, self.height - 1, int(y))

    def ipfmap2pos(self, x: int, y: int):
        # map range [0, width/ height] to range [0, 1] first
        # x, y = x - self.width // 2, y - self.height // 2
        x, y = float(x), float(y)
        x, y = x / self.width, y / self.height
        x, y = x * 2, y * 2
        x, y = x - 1, y - 1
        return x, y

    def updateipf(self, world):

        all_poses = [entity.state.p_pos for entity in world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))
        self.cam_range = cam_range
        for e, entity in enumerate(world.entities):
            # geometry
            x, y = entity.state.p_pos
            x, y = self.pos2ipfmap(x, y)
            if isinstance(entity, Agent):
                self.ipfmap[0, int(x), int(y)] = MIN_POTENTIAL
            elif isinstance(entity, Landmark):
                self.ipfmap[0, int(x), int(y)] = MAX_POTENTIAL
            else:
                self.ipfmap[0, int(x), int(y)] = 0

        for i in range(len(self.ipfmap) - 1):
            self.ipfmap[i + 1, :, :] = numpy_conv(self.ipfmap[i, :, :], kernel)

    # def updateipf2(self, world):
    #
    #     # update bounds to center around agent
    #     all_poses = [entity.state.p_pos for entity in world.entities]
    #     cam_range = np.max(np.abs(np.array(all_poses)))
    #
    #     for e, entity in enumerate(world.entities):
    #         # geometry
    #         x, y = entity.state.p_pos
    #         y *= (
    #             -1
    #         )  # this makes the display mimic the old pyglet setup (ie. flips image)
    #         x = (
    #                 (x / cam_range) * self.width // 2 * 0.9
    #         )  # the .9 is just to keep entities from appearing "too" out-of-bounds
    #         y = (y / cam_range) * self.height // 2 * 0.9
    #         x += self.width // 2
    #         y += self.height // 2
    #         if isinstance(entity, Agent):
    #             self.ipfmap[0, int(x), int(y)] = MIN_POTENTIAL
    #         elif isinstance(entity, Landmark):
    #             self.ipfmap[0, int(x), int(y)] = MAX_POTENTIAL
    #         else:
    #             self.ipfmap[0, int(x), int(y)] = 0
    #
    #     for i in range(len(self.ipfmap) - 1):
    #         x = 0
    #         for y in range(1, self.height - 1):
    #             self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x:x + 2, y - 1:y + 2]))
    #
    #         x = self.width - 1
    #         for y in range(1, self.width - 1):
    #             self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 1, y - 1:y + 2]))  # * 0.9
    #
    #         y = 0
    #         for x in range(1, self.width - 1):
    #             self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 2, y:y + 2]))
    #
    #         y = self.height - 1
    #         for x in range(1, self.width - 1):
    #             self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 2, y - 1:y]))
    #
    #         for x in range(1, self.width - 1):
    #             for y in range(1, self.width - 1):
    #                 self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 2, y - 1:y + 2]))

    # TODO: CHANGE THIS

    def reward(self, agent: Agent, world):
        x, y = self.pos2ipfmap(agent.state.p_pos[0], agent.state.p_pos[1])
        return self.ipfmap[self.iter_count - 1, x, y]

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
