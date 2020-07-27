import time
import numpy as np
import sys
import math
from itertools import combinations
from xml.etree.ElementTree import parse
from scipy.spatial.distance import pdist, cdist

# import ext_state

def make_env(name):
    if name == "basic":
        return Environment('./envs/basic_env.xml')
    elif name == "basic2":
        return Environment('./envs/basic_env_2.xml')
    elif name == "obstacles":
        return Environment('./envs/obstacles.xml')
    elif name == "crossing":
        return Environment('./envs/crossing.xml')
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

        self.n_ray = 20 # number of rays
        self.d = 4 # depth map size

        self.avg_times = np.zeros(10, dtype=np.float64)

        # self.reset()
        self.shape_internal = 3
        self.shape_external = (self.d, self.n_ray)
        # self.num_observation = 3 + self.n_ray * self.d # number of states
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
            
        # for vectorization
        self.p_t = np.array([agent.pos for agent in self.agents]) # positions
        self.v_t = np.array([agent.vel for agent in self.agents]) # velocities
        self.w_t = np.array([agent.theta for agent in self.agents]) # angles
        self.o_t = np.array([agent.ori for agent in self.agents]) # orientations
        self.targets = np.array([agent.target for agent in self.agents])
        self.obs_pos = np.array([obstacle.pos for obstacle in self.obstacles])
        self.p_t1 = self.v_t1 = self.w_t1 = self.o_t1 = None

        self.frame = 1

        ext_state = self.externalStates()
        depth_maps = []
        for i in range(len(self.agents)):
            depth_maps.append([ext_state[i]] * self.d)
        self.depth_maps = np.array(depth_maps)

        return self.computeStates()

    # action : [len(agents), 2] force
    def step(self, action):
        # 1. apply forces
        # start = time.perf_counter()
        force = action.reshape(len(self.agents), 2, 1)
        self.updateStates(force)
        # elapsed = time.perf_counter() - start
        # self.avg_times[0] += (elapsed - self.avg_times[0]) / self.frame

        # 2. compute rewards
        # start = time.perf_counter()
        rewards = self.computeRewards()
        # elapsed = time.perf_counter() - start
        # self.avg_times[1] += (elapsed - self.avg_times[1]) / self.frame

        self.p_t = self.p_t1
        self.v_t = self.v_t1
        self.w_t = self.w_t1
        self.o_t = self.o_t1

        # 3. compute states
        # start = time.perf_counter()
        states = self.computeStates()
        # elapsed = time.perf_counter() - start
        # self.avg_times[2] += (elapsed - self.avg_times[2]) / self.frame

        dist = np.linalg.norm(self.targets - self.p_t1, axis=1)
        dones = dist < self.eps

        self.frame += 1
        
        # print('update:', self.avg_times[0])
        # print('compute reward:', self.avg_times[1])
        # print('compute state:', self.avg_times[2])
        # print('internal state:', self.avg_times[4])
        # print('external state:', self.avg_times[5])

        return states, rewards, dones
    
    def updateStates(self, force):
        # update positions, velocities
        n = len(self.agents)
        self.p_t1 = self.p_t + self.dt * self.v_t
        self.v_t1 = self.v_t + self.dt * (self.o_t @ force).reshape(n, 2)

        # update orientations
        def rotationMatrix(c, s):
            return np.array([[c, -s], [s, c]])

        self.w_t1 = np.arctan2(self.v_t1[:, 1], self.v_t1[:, 0])
        cos = np.cos(self.w_t1)
        sin = np.sin(self.w_t1)

        for i in range(len(self.agents)):
            self.agents[i].pos = self.p_t1[i]
            self.agents[i].vel = self.v_t1[i]
            self.agents[i].theta = self.w_t1[i]
            self.agents[i].ori = rotationMatrix(cos[i], sin[i])

        self.o_t1 = [agent.ori for agent in self.agents]

    def computeRewards(self):
        # 1. distance reward(continuous)
        dist1 = np.linalg.norm(self.targets - self.p_t, axis=1)
        dist2 = np.linalg.norm(self.targets - self.p_t1, axis=1)
        distance_reward = dist1 - dist2

        # discrete distance reward
        # distance_reward = np.zeros(len(self.agents), dtype=np.float64)
        # distance_reward[dist < self.eps] = 1

        n = len(self.agents)

        # 2. collision reward
        min_collision_reward = -10
        collision_reward = np.zeros(n, dtype=np.float64)
        dist = pdist(self.p_t1)
        indices = list(combinations(range(n), 2))
        for k in range(len(indices)):
            i, j = indices[k]
            a = self.agents[i]
            b = self.agents[j]
            d = (dist[k] - (a.r + b.r)) ** 2
            r = max([min_collision_reward, -1 / (d + 1e-8)])
            collision_reward[i] += r
            collision_reward[j] += r
        
        if len(self.obstacles) > 0:
            dist = cdist(self.p_t1, self.obs_pos)
        for i in range(n):
            for j in range(len(self.obstacles)):
                a = self.agents[i]
                b = self.obstacles[j]
                d = (dist[i][j] - (a.r + b.r)) ** 2
                r = max([-1, -1 / (d + 1e-8)])
                collision_reward[i] += r

        # non-scipy version
        # collision_reward = np.zeros(len(self.agents), dtype=np.float64)
        # for i in range(len(self.agents)):
        #     for j in range(i + 1, len(self.agents)):
        #         a = self.agents[i]
        #         b = self.agents[j]
        #         d = self.distance(a.pos, b.pos) - (a.r + b.r)
        #         r = max([-1, -1 / (d + 1e-8)])
        #         collision_reward[i] += r
        #         collision_reward[j] += r
        #
        # for i in range(len(self.agents)):
        #     for j in range(len(self.obstacles)):
        #         a = self.agents[i]
        #         b = self.obstacles[j]
        #         d = self.distance(a.pos, b.pos) - (a.r + b.r)
        #         r = max([-1, -1 / (d + 1e-8)])
        #         collision_reward[i] += r
        #         collision_reward[i] += reward

        velocity_reward = np.zeros(n, dtype=np.float64)
        # velocity = np.linalg.norm(new_vel, axis=1)
        for i in range(n):
            v = np.sqrt(np.dot(self.agents[i].vel, self.agents[i].vel))
            velocity_reward[i] = -self.flood(v, self.v_min, self.v_max)

        orientation_reward = np.zeros(n, dtype=np.float64)
        ori_diff = np.abs(self.w_t1 - self.w_t)
        for i in range(n):
            orientation_reward[i] = -self.flood(ori_diff[i], self.w_min, self.w_max)

        if self.frame % 500 == 0:
            print('frame:', self.frame)
            print('distance_reward:', distance_reward)
            print('collision_reward:', collision_reward)
            print('velocity_reward:', velocity_reward)
            print('orientation reward:', orientation_reward)
            print()
        
        return self.w1 * distance_reward + self.w2 * collision_reward + self.w3 * velocity_reward + self.w4 * orientation_reward

    def computeStates(self):
        # start = time.perf_counter()
        int_state = self.internalStates()
        # elapsed = time.perf_counter() - start
        # self.avg_times[4] += (elapsed - self.avg_times[4]) / self.frame

        # start = time.perf_counter()
        ext_state = self.externalStates()
        # elapsed = time.perf_counter() - start
        # self.avg_times[5] += (elapsed - self.avg_times[5]) / self.frame

        for i in reversed(range(1, self.d)):
            self.depth_maps[:,i,:] = self.depth_maps[:,i-1,:]
        self.depth_maps[:,0,:] = ext_state

        return int_state, self.depth_maps
        
        return np.concatenate((int_state, self.depth_maps.reshape(-1, self.n_ray * self.d)), axis=1)

    # states : list of [pos, |vel|] -> [len(agents), 3]
    def internalStates(self):
        n = len(self.agents)
        m_inv = np.linalg.inv(self.o_t)
        pos_diff = (self.targets - self.p_t).reshape(n, 2, 1)
        relative_pos = (m_inv @ pos_diff).reshape(n, 2)
        v = np.linalg.norm(self.v_t, axis=1).reshape(n, 1)
        state_int = np.concatenate((relative_pos, v), axis=1)
        return state_int

    def externalStates(self):
        n_agent = len(self.agents)
        dw = np.array([(i*np.pi) / (self.n_ray-1) - np.pi/2 for i in range(self.n_ray)])
        thetas = np.array([w + dw for w in self.w_t])
        d = np.empty((n_agent, self.n_ray, 2))
        d[:,:,0] = np.cos(thetas)
        d[:,:,1] = np.sin(thetas)

        objs = np.concatenate((self.agents, self.obstacles))
        if len(self.obstacles) > 0:
            obj_pos = np.concatenate((self.p_t, self.obs_pos), axis=0)
        else:
            obj_pos = self.p_t
        p = np.array([[obj - agent for obj in obj_pos] for agent in self.p_t])

        state_ext=[]
        for i in range(n_agent):
            depth_map = []
            for j in range(self.n_ray):
                dij = d[i][j]
                t_min = self.t_max
                for k in range(len(obj_pos)):
                    if i == k:
                        continue
                    pik = p[i][k]
                    tm = np.dot(pik, dij)
                    lm_2 = np.dot(pik, pik) - tm ** 2
                    dt = objs[k].r ** 2 - lm_2
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

        # -- Cython --
        # r = np.array([obj.r for obj in objs])
        # state_ext = ext_state.external_states(d, p, r, n_agent, self.n_ray, objs.shape[0], self.t_max)
        # state_ext = np.array(state_ext)
        # print(state_ext.shape)

        # return state_ext

        # -- No vectorization --
        # state_ext = []
        # for agent in self.agents:
        #     depth_map = []
        #     for i in range(self.n_ray):
        #         theta = agent.theta + (i * np.pi) / (self.n_ray - 1) - np.pi / 2
        #         d = np.array([math.cos(theta), math.sin(theta)])
        #         objs = np.concatenate((self.agents, self.obstacles))
        #         t_min = self.t_max
                
        #         for obj in objs:
        #             if agent == obj:
        #                 continue
        #             p = obj.pos - agent.pos
        #             tm = np.dot(p, d)
        #             lm_2 = np.dot(p, p) - tm ** 2
        #             dt = obj.r ** 2 - lm_2
        #             if dt > 0:
        #                 dt = np.sqrt(dt)
        #                 t0 = tm - dt
        #                 t1 = tm + dt
        #                 if t0 > 0:
        #                     t_min = min(t_min, t0)
        #                 elif t1 > 0:
        #                     t_min = min(t_min, t1)
        #         depth_map.append(t_min)
        #     state_ext.append(depth_map)
        # return np.array(state_ext)

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def flood(self, value, v_min, v_max):
        return abs(np.min([value - v_min, 0])) + abs(max([value - v_max, 0]))

class Agent():

    def __init__(self, x, y, x1, y1, r):
        
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.target = np.array([x1, y1], dtype=np.float64)
        self.r = r

        direction = self.target - self.pos
        self.theta = math.atan2(direction[1], direction[0])
        cos = math.cos(self.theta)
        sin = math.sin(self.theta)
        self.ori = np.array([[cos, -sin], [sin, cos]])

class Obstacle():

    def __init__(self, x, y, r):
        self.pos = np.array([x, y], dtype=np.float64)
        self.r = r

if __name__ == "__main__":
    env = make_env('basic')
    state = env.reset()
    state_int = env.internalStates()
    state_ext = env.externalStates()
    state = np.concatenate((state_int, state_ext), axis=1)