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


class Scenario(BaseScenario):

    def make_world(self):
        self.iter_count = 20
        self.width = 700
        self.height = 700
        self.ipfmap = np.zeros((self.iter_count, self.width, self.height), dtype=float)

        world = World()
        # add agents
        world.agents = [Agent() for i in range(3)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
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
        self.updateipf()

    def pos2ipfmap(self, x, y):
        # pos in range[-1 , 1] to range [0, 1] first
        x += 1
        y += 1
        x /= 2
        y /= 2
        x, y = x * self.width, y * self.height
        return x//1, y//1

    def ipfmap2pos(self, x: int, y: int):
        # map range [0, width/ height] to range [0, 1] first
        x, y = float(x), float(y)
        x, y = x / self.width, y / self.height
        x, y = x * 2, y * 2
        x, y = x - 1, y - 1
        return x, y

    def updateipf(self, world):

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        for e, entity in enumerate(world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                    (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            if isinstance(entity, Agent):
                self.ipfmap[0, x // 1, y // 1] = MIN_POTENTIAL
            elif isinstance(entity, Landmark):
                self.ipfmap[0, x // 1, y // 1] = MAX_POTENTIAL
            else:
                self.ipfmap[0, x // 1, y // 1] = 0

        for i in range(len(self.ipfmap) - 1):
            x = 0
            for y in range(1, self.height - 1):
                self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x:x + 2, y - 1:y + 2]))

            x = self.width - 1
            for y in range(1, self.width - 1):
                self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 1, y - 1:y + 2]))  # * 0.9

            y = 0
            for x in range(1, self.width - 1):
                self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 2, y:y + 2]))

            y = self.height - 1
            for x in range(1, self.width - 1):
                self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 2, y - 1:y]))

            for x in range(1, self.width - 1):
                for y in range(1, self.width - 1):
                    self.ipfmap[i + 1, x, y] = (np.sum(self.ipfmap[i, x - 1:x + 2, y - 1:y + 2]))

    # TODO: CHANGE THIS
    def reward(self, agent:Agent, world):

        return self.ipfmap[self.iter_count-1,self.pos2ipfmap(agent.state.p_pos[0], agent.state.p_pos[1])]

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)