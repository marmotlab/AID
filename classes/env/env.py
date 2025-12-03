import copy

import numpy as np
from itertools import product

from classes.prm.Graph import Graph
from classes.prm.Obstacle import Obstacle
from classes.prm.PRMController import PRMController
from classes.env.Gaussian2D import Gaussian2D
from classes.env.gp_ipp import GaussianProcessForIPP


class Env():
    def __init__(self, num_agent, sample_size=500, measurement_interval=0.2, k_size=10, start=None,
                 destination=None, obstacle=[], save_image=False, seed=None, adaptive_area=True,
                 threshold=0.4, beta=1, sensor_range=0.45, gaussian_num=(8, 12)):
        # if seed:
        #     print(f"Env init seed: {seed}")
        #     np.random.seed(seed)

        # if start is None:
        #     start = np.random.rand(2)
        # if destination is None:
        #     destination = np.random.rand(2)

        if start:
            start = np.array(start)
        if destination:
            destination = np.array(destination)
            
        # Constants
        self.num_agent = num_agent
        self.measurement_interval = measurement_interval
        self.adaptive_area = adaptive_area
        self.threshold = threshold
        self.beta = beta
        self.sensor_range = sensor_range
        self.gaussian_num = gaussian_num

        # Environment variables
        self.reset_flag = False

        # PRM Generation parameters
        self.seed = seed
        self.sample_size = sample_size
        self.k_size = k_size
        self.start = start
        self.destination = destination
        self.obstacle = obstacle

        # PRM Graph data structure
        self.node_coords, self.graph, self.prm = dict(), dict(), dict()

        # underlying_distribution
        self.underlying_distribution = None
        self.ground_truth = None

        # GP
        self.gp_ipp = None

        # Evaluation variables
        self.cov_trace0 = None

        # Multi-agent data structure
        self.cov_trace = dict()
        self.dist_residual = dict()
        self.current_position = dict()
        self.node_info, self.node_std = dict(), dict()

        # Save image
        self.save_image = save_image
        self.frame_files = []


    def reset(self, seed=None):
        if seed:
            print(f"Env reset seed: {seed}")
            np.random.seed(seed)
        else:
            print(f"Env reset seed: {self.seed}")
            np.random.seed(self.seed)

        # Reset start and destination
        if self.start is None:
            self.start = np.random.rand(2)
        if self.destination is None:
            self.destination = np.random.rand(2)

        # Return if agent_id has already been reset
        initial_env_obs = dict()

        # underlying distribution
        self.underlying_distribution = Gaussian2D(self.gaussian_num)
        self.ground_truth = self.get_ground_truth()

        # initialize gp
        self.gp_ipp = GaussianProcessForIPP(self.threshold, self.beta, self.sensor_range)
        high_info_area = self.gp_ipp.get_high_info_area() if self.adaptive_area else None
        cov_trace = self.gp_ipp.evaluate_cov_trace(high_info_area)
        self.gp_ipp.update_gp()

        # Evaluation variables
        self.cov_trace0 = cov_trace

        # initialize common variables
        for i in range(self.num_agent):
            self.cov_trace[f"{i}"] = cov_trace
            self.dist_residual[f"{i}"] = 0
            self.current_position[f"{i}"] = self.start
        
        ## Once per agent_id
        # PRM Generation (Different PRM for each agent but same start and destination)
        coordinates = np.random.rand(self.sample_size, 2) # between [0,1)
        
        for agent_id in range(self.num_agent):
            self.prm[f"{agent_id}"] = PRMController(self.sample_size, self.obstacle, self.start,
                                                self.destination, self.k_size)
        
            agent_node_coords, agent_graph = self.prm[f"{agent_id}"].runPRM(saveImage=False,
                                                                        seed=self.seed,
                                                                        start_pos=self.start,
                                                                        coordinates=coordinates)
            # NOTE np array shape (sample_size+2, 2), 0 is destination, 1 is start
            self.node_coords[f"{agent_id}"] = agent_node_coords 
            self.graph[f"{agent_id}"] = agent_graph
            self.node_info[f"{agent_id}"], self.node_std[f"{agent_id}"] =\
                self.gp_ipp.update_node(self.node_coords[f"{agent_id}"])

        initial_env_obs["node_coords"] = self.node_coords # only in initial_env_obs
        initial_env_obs["graph"] = self.graph # only in initial_env_obs
        initial_env_obs["node_info"] = self.node_info
        initial_env_obs["node_std"] = self.node_std
        initial_env_obs["cov_trace"] = cov_trace
        initial_env_obs["current_position"] = self.start
        initial_env_obs["gp_ipp"] = copy.deepcopy(self.gp_ipp)

        self.reset_flag = True
        return initial_env_obs
    
    def step_coord(self, agent_id, next_coord, route, done, lock, measurement=True):
        obs = dict()
        gp_ipp_obs = None

        next_coord = next_coord
        current_coord = self.current_position[f"{agent_id}"]

        dist = np.linalg.norm(current_coord - next_coord)
        remain_length = dist
        next_length = self.measurement_interval - self.dist_residual[f"{agent_id}"]
        no_sample = True

        # for small tolerance due to float precision
        while remain_length > next_length - 1e-6:
            if no_sample:
                sample = (next_coord - current_coord) * next_length / dist + current_coord
            else:
                sample = (next_coord - current_coord) * next_length / dist + sample
            if measurement:
                observed_value = self.underlying_distribution.distribution_function(
                    sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
            else:
                observed_value = np.array([0])
            
            with lock: # lock as appending to lists in gp_ipp (not thread safe)
                self.gp_ipp.add_observed_point(sample, observed_value)
            
            remain_length -= next_length
            next_length = self.measurement_interval
            no_sample = False
        
        # Update GP
        with lock: # lock as update_gp, update_node, evaluate_cov_trace, get_high_info_area in gp_ipp (not thread safe)
            self.gp_ipp.update_gp()
            self.node_info[f"{agent_id}"], self.node_std[f"{agent_id}"] = self.gp_ipp.update_node(self.node_coords[f"{agent_id}"])

            if measurement:
                high_info_area = self.gp_ipp.get_high_info_area() if self.adaptive_area else None
                # eval stuff here
            
            new_cov_trace = self.gp_ipp.evaluate_cov_trace(high_info_area)
            gp_ipp_obs = copy.deepcopy(self.gp_ipp)

        # Rewards
        reward = 0
        # if next_coord in route[-2:]: # - reward for going back to the same node
        #     reward += -0.1
        # elif self.cov_trace[f"{agent_id}"] > new_cov_trace: # + reward for decreasing cov_trace
        #     reward += (self.cov_trace[f"{agent_id}"] - new_cov_trace) * 2 / self.cov_trace[f"{agent_id}"]
        # if done: # - reward for remaining cov_trace
        #     reward -= new_cov_trace / 50

        ## Original Reward
        # reward += (self.cov_trace[f"{agent_id}"] - new_cov_trace) / self.cov_trace[f"{agent_id}"]
        # if done:
        #     reward -= new_cov_trace / 20

        ## My reward
        if done:
            reward = -5 * (new_cov_trace / self.cov_trace0) ** 0.5
        
        # Update variables
        self.cov_trace[f"{agent_id}"]= new_cov_trace
        self.current_position[f"{agent_id}"] = next_coord
        self.dist_residual[f"{agent_id}"] = self.dist_residual[f"{agent_id}"] + remain_length if no_sample else remain_length
        
        # Update return observation
        obs["node_info"] = self.node_info[f"{agent_id}"]
        obs["node_std"] = self.node_std[f"{agent_id}"]
        obs["cov_trace"] = self.cov_trace[f"{agent_id}"]
        obs["current_position"] = next_coord
        obs["gp_ipp"] = gp_ipp_obs
        return reward, obs

    def is_reset(self):
        return self.reset_flag

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth
    
    # def plot(self, agent_id):
    #     self.gp_ipp.plot(self.ground_truth)
    #     plt.scatter(self.node_coords[f"{agent_id}"][1][0], self.node_coords[f"{agent_id}"][1][1], c='r', marker='*',
    #         s=15 ** 2)
        
    #     # Predicted interesting area subplot (high info area)
    #     plt.subplot(2, 3, 5)
    #     plt.title('Interesting area')
    #     x = self.high_info_area[:, 0]
    #     y = self.high_info_area[:, 1]
    #     plt.hist2d(x, y, bins=30, vmin=0, vmax=1, edgecolors="face")

    # def plot(self, agent_id, path, agent_route, budget):
    #     colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
    #     agent_color = colorlist[agent_id]
    #     agent_step = len(agent_route) - 1
    #     cov_trace = self.cov_trace[f"{agent_id}"]
    #     # budget = self.budget[f"{agent_id}"]

    #     # print(f"Agent {agent_id}: Plotting Step {agent_step}")

    #     plt.switch_backend('agg')
    #     self.gp_ipp.plot(self.ground_truth)

    #     # Agent Location and Path on mean prediction subplot
    #     plt.scatter(self.node_coords[f"{agent_id}"][1][0], self.node_coords[f"{agent_id}"][1][1], c='r', marker='*',
    #                 s=15 ** 2)
        
    #     for id in range(self.num_agent):
    #         pointsToDisplay = [(self.prm[f"{id}"].findPointsFromNode(path)) for path in self.agent_route[f"{id}"]]
    #         x = [item[0] for item in pointsToDisplay]
    #         y = [item[1] for item in pointsToDisplay]
    #         for i in range(len(x) - 1):
    #             plt.plot(x[i:i + 2], y[i:i + 2], c=colorlist[id], linewidth=4, zorder=5,
    #                     alpha=0.25 + 0.6 * i / len(x))
        
    #     # Predicted interesting area subplot (high info area)
    #     plt.subplot(2, 3, 5)
    #     plt.title('Interesting area')
    #     x = self.high_info_area[:, 0]
    #     y = self.high_info_area[:, 1]
    #     plt.hist2d(x, y, bins=30, vmin=0, vmax=1, edgecolors="face")

    #     # Intent map subplot
    #     plt.suptitle('Color: {} Cov trace: {:.4g}  remain_budget: {:.4g}'.format(agent_color, cov_trace, budget))
    #     plt.savefig(f'{path}/agent{agent_id}_step{agent_step}.png', dpi=150)
    #     plt.cla()
    #     plt.close("all")


if __name__ == '__main__':
    env = Env()
    env.reset(0)