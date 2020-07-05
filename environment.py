import time
import numpy as np
import sys
import math
from xml.etree.ElementTree import parse

from util import *

def make_env(name):
    if name == "basic":
        return Environment('./envs/basic_env.xml')
    if name == "obstacles":
        return Environment('./envs/obstacles.xml')
    return None

class Environment():

    def __init__(self, env_fname):
        self.tree = parse(env_fname) 

        self.eps = 1 # distance between agent and target
        self.dt = 1e-1 # timestep

        self.v_min = 0.0 # desired minimum velocity
        self.v_max = 1.5 # desired maximum velocity
        self.t_max = 40 # maximum sight of agent

        self.w_min = 0.0 # desired minimum angular delta
        self.w_max = np.pi / 4 # desired maximum angular delta
        
        self.w1 = 4.0 # distance reward weight
        self.w2 = 3.0 # collision reward weight
        self.w3 = 2.0 # velocity flood reward weight
        self.w4 = 1.0 # theta flood reward weight

        self.n = 20 # number of rays

        self.reset()
        self.num_observation = 3 + self.n # number of states
        self.num_action = 2 # number of actions
        
    def reset(self):
        root = self.tree.getroot()

        self.agents = []
        for agent in root.findall('agent'):
            x = float(agent.findtext('x'))
            y = float(agent.findtext('y'))
            x1 = float(agent.findtext('x1'))
            y1 = float(agent.findtext('y1'))
            r = float(agent.findtext('radius'))
            self.agents.append(Agent(x, y, x1, y1, r))

        self.obstacles = []
        for obstacle in root.findall('obstacle'):
            x = float(obstacle.findtext('x'))
            y = float(obstacle.findtext('y'))
            r = float(obstacle.findtext('radius'))
            self.obstacles.append(Obstacle(x, y, r))
            
        self.frame = 0

        return self.computeStates()

    # action : [len(agents), 2] force
    def step(self, action):
        p_t = np.array([agent.pos for agent in self.agents]) # positions
        v_t = np.array([agent.vel for agent in self.agents]) # velocities
        w_t = np.array([agent.theta for agent in self.agents]) # thetas
        o_t = np.array([agent.ori for agent in self.agents]) # orientations
        
        action = action.reshape(len(self.agents), 2, 1)
        p_t1, v_t1, w_t1 = self.updateStates(p_t, v_t, o_t, action)
        
        targets = np.array([agent.target for agent in self.agents])
        dist = np.linalg.norm(targets - p_t1, axis=1)

        states = self.computeStates()
        rewards = self.computeRewards(p_t, p_t1, w_t, w_t1)
        dones = dist < self.eps

        self.frame += 1

        return states, rewards, dones
    
    def updateStates(self, pos, vel, ori, force):
        # update positions, velocities
        n = len(self.agents)
        new_pos = pos + self.dt * vel
        new_vel = vel + self.dt * (ori @ force).reshape(n, 2)

        # update orientations
        def rotationMatrix(c, s):
            return np.array([[c, -s], [s, c]])

        new_theta = np.arctan2(new_vel[:, 1], new_vel[:, 0])
        cos = np.cos(new_theta)
        sin = np.sin(new_theta)

        for i in range(len(self.agents)):
            self.agents[i].pos = new_pos[i]
            self.agents[i].vel = new_vel[i]
            self.agents[i].theta = new_theta[i]
            self.agents[i].ori = rotationMatrix(cos[i], sin[i])
        
        return new_pos, new_vel, new_theta

    def computeRewards(self, pos, new_pos, theta, new_theta):
        targets = np.array([agent.target for agent in self.agents])

        # 1. distance reward
        dist1 = np.linalg.norm(targets - pos, axis=1)
        dist2 = np.linalg.norm(targets - new_pos, axis=1)
        distance_reward = dist1 - dist2
        # distance_reward = np.zeros(len(self.agents)).astype(np.float64)
        # distance_reward[dist < self.eps] = 1

        # 2. collision reward
        collision_reward = np.zeros(len(self.agents)).astype(np.float64)
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                a = self.agents[i]
                b = self.agents[j]
                dist = self.distance(a.pos, b.pos)
                if dist > a.r + b.r:
                    dist -= (a.r + b.r) / 2
                    reward = -1 / (dist + 1e-8)
                else:
                    reward = -self.w2
                collision_reward[i] += reward
                collision_reward[j] += reward

        for i in range(len(self.agents)):
            for j in range(len(self.obstacles)):
                a = self.agents[i]
                b = self.obstacles[j]
                dist = self.distance(a.pos, b.pos)
                if dist > a.r + b.r:
                    dist -= (a.r + b.r) / 2
                    reward = -1 / (dist + 1e-8)
                else:
                    reward - self.w2
                collision_reward[i] += reward

        velocity_reward = np.zeros(len(self.agents)).astype(np.float64)
        for i in range(len(self.agents)):
            v = np.sqrt(np.dot(self.agents[i].vel, self.agents[i].vel))
            velocity_reward[i] = -self.flood(v, self.v_min, self.v_max)

        orientation_reward = np.zeros(len(self.agents)).astype(np.float64)
        ori_diff = np.abs(new_theta - theta)
        for i in range(len(self.agents)):
            orientation_reward[i] = -self.flood(ori_diff[i], self.w_min, self.w_max)

        if self.frame % 500 == 0:
            print('frame:', self.frame)
            print('ori_diff:', ori_diff)
            print('distance_reward:', distance_reward)
            print('collision_reward:', collision_reward)
            print('velocity_reward:', velocity_reward)
            print('orientation reward:', orientation_reward)
            print()
        
        return self.w1 * distance_reward + self.w2 * collision_reward + self.w3 * velocity_reward + self.w4 * orientation_reward

    def computeStates(self):
        state_int = self.internalStates()
        state_ext = self.externalStates()
        return np.concatenate((state_int, state_ext), axis=1)

    # states : array of [pos, vel] -> [len(agents), 4]
    def internalStates(self):
        state_int = []
        for agent in self.agents:
            m_inv = np.linalg.inv(agent.ori)
            pos_diff = (agent.target - agent.pos).reshape(2, 1)
            relative_pos = (m_inv @ pos_diff).reshape(2)
            v = np.sqrt(np.dot(agent.vel, agent.vel))
            state = np.array([relative_pos[0], relative_pos[1], v])
            state_int.append(state)
        return np.array(state_int)

    def externalStates(self):
        state_ext = []
        for agent in self.agents:
            depth_map = []
            for i in range(self.n):
                theta = agent.theta + (i * np.pi) / (self.n - 1) - np.pi / 2
                d = np.array([math.cos(theta), math.sin(theta)])
                objs = np.concatenate((self.agents, self.obstacles))
                t_min = self.t_max
                
                for obj in objs:
                    if agent == obj:
                        continue
                    p = obj.pos - agent.pos
                    tm = np.dot(p, d)
                    lm_2 = np.dot(p, p) - tm ** 2
                    dt = obj.r ** 2 - lm_2
                    if dt > 0:
                        dt = np.sqrt(dt)
                        t0 = tm - dt
                        t1 = tm + dt
                        if t0 > 0:
                            t_min = min(t_min, t0)
                        elif t1 > 0:
                            t_min = min(t_min, t1)
                depth_map.append(t_min)
            state_ext.append(depth_map)
        return np.array(state_ext)

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def flood(self, value, v_min, v_max):
        return abs(np.min([value - v_min, 0])) + abs(max([value - v_max, 0]))

class Agent():

    def __init__(self, x, y, x1, y1, r):
        
        self.pos = np.array([x, y]).astype(np.float64)
        self.vel = np.zeros(2).astype(np.float64)
        self.target = np.array([x1, y1]).astype(np.float64)
        self.r = r

        direction = self.target - self.pos
        self.theta = math.atan2(direction[1], direction[0])
        cos = math.cos(self.theta)
        sin = math.sin(self.theta)
        self.ori = np.array([[cos, -sin], [sin, cos]])

        self.vecComputeOrientation = np.vectorize(self.computeOrientation)
    
    def computeOrientation(self):
        self.theta = math.atan2(self.vel[1], self.vel[0])
        cos = math.cos(self.theta)
        sin = math.sin(self.theta)
        self.ori = np.array([[cos, -sin], [sin, cos]])
    


    def to_numpy(self):
        return np.array([self.pos, self.vel])

    def __str__(self):
        return str(self.pos)

class Obstacle():

    def __init__(self, x, y, r):
        self.pos = np.array([x, y]).astype(np.float64)
        self.r = r

if __name__ == "__main__":
    env = make_env('basic')
    state = env.reset()
    state_int = env.internalStates()
    state_ext = env.externalStates()
    state = np.concatenate((state_int, state_ext), axis=1)
    
    from model import ActorCritic

    for i in range(10):
        action = np.array([[1., 1.], [1., 1.], [1., 1.]])
        print('action:', action)
        state, reward, done, = env.step(action)

        print('state:', state)

    # print('state_int:', state_int.shape)
    # print(state_int)
    # print('state_ext:', state_ext.shape)
    # print('state:', state.shape)
    # print(state)