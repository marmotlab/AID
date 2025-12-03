import copy

import numpy as np

from classes.RIG_original import RRT
from classes.util.predictor import Predictor


class Agent:
    def __init__(self):
        pass

    def act(self, obs):
        raise NotImplementedError

'''
NOTE:
- shared_memory is from runner
- Predict measurement is dict for claimed trajectory of each agent
- Env is step in a loop for all nodes in the trajectory OR add step traj in Env
- Does not do sample number or store samples, just copy gp_ipp from env
'''

# Agent to take in obs and return an action and calculate if its done based on budget
class RIGTreeAgent(Agent):
    def __init__(self, agent_id, shared_memory, num_agent, measurement_interval, budget,
                 radius, branch_length, rrt_iterations, action_horizon,
                 trajectory_length, trajectory_intent):
        self.agent_id = agent_id
        self.shared_memory = shared_memory
        self.num_agent = num_agent
        self.measurement_interval = measurement_interval
        self.budget = budget # Budget of agent

        # Agent Variables
        self.budget_remaining = None # Remaining budget of agent
        self.current_position = None
        self.dist_residual = None # Distance remaining to next measurement point
        self.node_coords = None

        # RIGTree Parameters
        self.radius = radius # radius for RRT 0.5
        self.measurement_interval = measurement_interval # distance between measurements 0.2
        self.branch_length = branch_length # step size for RRT 0.1
        self.rrt_iterations = rrt_iterations # Number of nodes in the tree 400
        self.action_horizon = action_horizon # portion of the trajectory executed by the agent 0.21
        self.trajectory_length = trajectory_length # length of trajectory [0.9, 1.0]
        self.trajectory_intent = trajectory_intent # True use predicted traj for SGA, False use only executed portion

    def reset(self, initial_env_obs):
        self.budget_remaining = self.budget
        self.current_position = initial_env_obs["current_position"] # np.array shape (2,)
        self.dist_residual = 0
        self.node_coords = initial_env_obs["node_coords"][f"{self.agent_id}"]

    def plan(self, obs):
        # Construct RRT Tree
        # Sample point
        # Find Closest Node
        # New Node (step from closest node to sampled point)
        # Check if new node is valid
        # Find near nodes to new node
        # for each near node, 
        #   step from near node to new node
        #   if valid, calculate info and cost
        #   if check_prune, prune
        #   else add to tree
        #   if cost of new node is more than budget, add to closed list

        self.current_position = obs["current_position"]
        node_coords = np.array([[]])

        gp_ipp_copy = obs["gp_ipp"] # take from obs

        self.rrt = RRT(num_nodes=20, XDIM=1.0, YDIM=1.0, radius=self.radius,
                       branch_length=self.branch_length, gp_func=gp_ipp_copy, gaussian_distrib=None)
        nodes = self.rrt.RRT_planner(start_node=self.current_position,
                                     iterations=self.rrt_iterations, info=None)

        for each_node in nodes:
            append_it = np.array([[each_node.x, each_node.y]])
            node_coords = np.array(np.append(node_coords, append_it))

        node_coords = node_coords.reshape(-1, 2)

        # Virtual Measurements are planned trajectory of other agents
        # Predicted Measurements is shared memory
        virtual_measurements = []
        for i in range(self.num_agent):
            if i != int(self.agent_id):
                for measurement in self.shared_memory["predict_measurements"][f"{i}"]:
                    virtual_measurements.append(measurement)
    
        predictor = Predictor(self.agent_id, gp_ipp_copy, virtual_measurements, self.measurement_interval)

        path = []
        destination_list = [] # List of Nodes within trajectory length
        best_trajectory_node = []
        best_trajectory = []
        all_trajectory = []

        # Get all nodes within trajectory length
        for node in nodes:
            if self.trajectory_length[0] < node.cost < self.trajectory_length[1]:
                destination_list.append(node)

        best_cov_trace = float("infinity")
        new_cov_trace = float("infinity")

        # Loop through all possible trajectories
        for destination_node in destination_list:
            trajectory = [np.array([destination_node.x, destination_node.y])]
            trajectory_node = [destination_node]
            while destination_node.parent.parent != None:
                destination_node = destination_node.parent
                trajectory.insert(0, np.array([destination_node.x, destination_node.y]))
                trajectory_node.insert(0, destination_node)
            all_trajectory.append(trajectory)
            dist_residual_prediction = self.dist_residual # init to 0
            predict_measurements = []
            current_coord = self.current_position
            predictor_copy = copy.deepcopy(predictor)
            for next_coord in trajectory:
                new_cov_trace, dist, dist_residual_prediction, predict_measurements = \
                    predictor_copy.predict_step(next_coord,
                                                current_coord,
                                                dist_residual_prediction,
                                                predict_measurements)

                current_coord = next_coord
            if new_cov_trace < best_cov_trace:
                best_cov_trace = new_cov_trace
                best_trajectory_node = copy.deepcopy(trajectory_node)
                best_trajectory = copy.deepcopy(trajectory)

        # This is to get the actual portion executed by the agent
        for node in best_trajectory_node:
            if node.cost < self.action_horizon:
                path.append(np.array([node.x, node.y]))
            else:
                break
        
        planned_trajectory = best_trajectory
        # if USE_PLOT:
        #     self.rrt.plot(trajectory=best_trajectory, prior_position=self.global_agent_pos[f"{agent_ID}"],
        #                   agent_ID=agent_ID, path_length=path_length, all_trajectory=all_trajectory)

        return path, planned_trajectory
        