import os
import time
import copy

import torch
import hydra
import imageio
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from collections import deque

from classes.util.utils import calculate_position_embedding, calculate_intent_mean_cov, calculate_intent_info
from classes.util.predictor import Predictor
plt.switch_backend('agg')


class RLMAIPPWorker:
    def __init__(self,
                 meta_agent_id, agent_id, episode_number, env, shared_memory, lock, num_agent,
                 model, weights, num_paths, sampling_steps, k_size, embedding_size,
                 budget, steps, eval=False, device='cpu', save_image=False, gifs_path='gifs'):
        self.meta_agent_id = meta_agent_id
        self.agent_id = agent_id
        self.episode_number = episode_number
        self.env = env
        self.shared_memory = shared_memory
        self.lock = lock
        self.num_agent = num_agent
        self.steps = steps
        self.device = device

        # For plotting
        self.save_image = save_image
        self.gifs_path = gifs_path
        self.frame_files = []

        self.episode_gifs_path = '{}/{}'.format(self.gifs_path, self.episode_number)
        if not os.path.exists(self.episode_gifs_path) and self.save_image:
            os.makedirs(self.episode_gifs_path)

        # For Agent
        self.model = hydra.utils.instantiate(model)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.eval_mode = eval
        if eval:
            self.model.eval()
        self.num_paths = num_paths
        self.sampling_steps = sampling_steps
        self.k_size = k_size
        self.embedding_size = embedding_size

        self.budget = budget
        self.budget_remaining = self.budget
        self.node_coords = None
        self.current_position = None

        # For Data Collection
        self.perf_metrics = None
        self.episode_data = dict()

    def work(self):
        self.perf_metrics = self.run_episode()
    
    def run_episode(self):
        # Init Data Collection
        node_inputs_data = []
        edge_inputs_data = []
        budget_inputs_data = []
        current_index_inputs_data = []
        lstm_h_inputs_data = []
        lstm_c_inputs_data = []
        pos_embedding_inputs_data = []
        actions_data = []
        terminated_data = []
        reward_data = []

        # First thread will reset the environment and shared memory
        with self.lock:
            if not self.env.is_reset():
                initial_env_obs = self.env.reset()
                self.shared_memory["initial_env_obs"] = initial_env_obs

            if not self.shared_memory["reset"]:
                # Initialize method specific shared memory
                self.shared_memory["intent_mean"] = dict()
                self.shared_memory["intent_std"] = dict()
                for agent_id in range(self.num_agent):
                    self.shared_memory["intent_mean"][f"{agent_id}"] = []
                    self.shared_memory["intent_std"][f"{agent_id}"] = []
    
                # Update shared memory
                self.shared_memory["reset"] = True

        # Reset agent and update shared memory with initial env obs
        self.node_coords = self.shared_memory["initial_env_obs"]["node_coords"][f"{self.agent_id}"]
        self.current_position = self.shared_memory["initial_env_obs"]["current_position"]

        # Make obs from initial_env_obs
        obs = copy.deepcopy(self.shared_memory["initial_env_obs"])
        obs["node_info"] = obs["node_info"][f"{self.agent_id}"]
        obs["node_std"] = obs["node_std"][f"{self.agent_id}"]

        # Update shared memory
        self.shared_memory["agent_position"][f"{self.agent_id}"] = obs["current_position"]
        self.shared_memory["agent_route"][f"{self.agent_id}"] = [obs["current_position"]]
        self.shared_memory["all_reset"][f"{self.agent_id}"] = True

        # Init some data
        graph = list(obs["graph"][f"{self.agent_id}"].values())
        node_edges = []
        for node in graph:
            edges = list(map(int, node))
            node_edges.append(edges) # (num_nodes, max_num_edges)
        pos_embedding = calculate_position_embedding(node_edges) # (num_nodes, 32)

        ## Wait for other agents to reset
        while not self.is_reset_all():
            time.sleep(0.5)

        # Get intent from shared memory
        with self.lock:
            gaussian_mean = copy.deepcopy(self.shared_memory["intent_mean"])
            gaussian_cov = copy.deepcopy(self.shared_memory["intent_std"])

        if self.save_image:
            self.plot(obs, gaussian_mean, gaussian_cov, planned_trajectory=[])

        # Init input
        self.node_inputs = None
        self.edge_inputs = None
        self.budget_inputs = None
        self.current_index_inputs = None
        self.lstm_h_inputs = torch.zeros((1, 1, self.embedding_size)).to(self.device)
        self.lstm_c_inputs = torch.zeros((1, 1, self.embedding_size)).to(self.device)
        self.pos_embedding_inputs = None

        for step in range(self.steps):
            # Construct current input
            self.node_inputs, self.edge_inputs, self.budget_inputs, \
                self.current_index_inputs, self.pos_embedding_inputs =\
                self.construct_input(self.node_coords, node_edges, pos_embedding,
                                    obs, gaussian_mean, gaussian_cov,
                                    self.budget_remaining)

            # Init path predictor
            predictor = Predictor(self.agent_id, obs["gp_ipp"], [], self.env.measurement_interval)

            best_cov_trace = float("infinity")
            new_cov_trace = float("infinity")
            best_path_id = None
            planned_path = [[] for _ in range(self.num_paths)]
            first_action_index_list = []
            first_next_node_index_list = []
            first_lstm_h_list = []
            first_lstm_c_list = []
            
            # Sampling planning
            for path_id in range(self.num_paths):
                dist_residual_prediction = 0
                predict_measurements = []
                current_coord_sampling = copy.deepcopy(self.current_position)
                predictor_copy = copy.deepcopy(predictor)

                # Copy current inputs for sampling path
                node_inputs_sampling = copy.deepcopy(self.node_inputs)
                edge_inputs_sampling = copy.deepcopy(self.edge_inputs)
                budget_inputs_sampling = copy.deepcopy(self.budget_inputs)
                current_index_inputs_sampling = copy.deepcopy(self.current_index_inputs)
                lstm_h_inputs_sampling = copy.deepcopy(self.lstm_h_inputs)
                lstm_c_inputs_sampling = copy.deepcopy(self.lstm_c_inputs)
                pos_embedding_inputs_sampling = copy.deepcopy(self.pos_embedding_inputs)

                # Copy current obs for sampling path
                remain_budget_sampling = copy.deepcopy(self.budget_remaining)
                obs_sampling = copy.deepcopy(obs)

                for sampling_step in range(self.sampling_steps):
                    with torch.no_grad():
                        logp_list_sampling, value_sampling, lstm_h_inputs_sampling, lstm_c_inputs_sampling = self.model(
                            node_inputs_sampling, edge_inputs_sampling,
                            budget_inputs_sampling,
                            current_index_inputs_sampling, lstm_h_inputs_sampling,
                            lstm_c_inputs_sampling, pos_embedding_inputs_sampling)
                        
                    action_index_sampling = torch.multinomial(logp_list_sampling.exp(), 1).long().squeeze(1)
                    next_node_index_sampling = edge_inputs_sampling[:, current_index_inputs_sampling.item(),
                                                action_index_sampling.item()]
                    
                    if sampling_step == 0:
                        first_action_index_list.append(action_index_sampling)
                        first_next_node_index_list.append(next_node_index_sampling)
                        first_lstm_h_list.append(lstm_h_inputs_sampling)
                        first_lstm_c_list.append(lstm_c_inputs_sampling)
                    next_coord_sampling = self.node_coords[next_node_index_sampling]
                    planned_path[path_id].append(next_coord_sampling)

                    # Predict step
                    new_cov_trace, dist, dist_residual_prediction, predict_measurements = \
                        predictor_copy.predict_step(next_coord_sampling,
                                                    current_coord_sampling,
                                                    dist_residual_prediction,
                                                    predict_measurements)
                    # Update obs for next sampling step
                    current_coord_sampling = next_coord_sampling
                    remain_budget_sampling -= dist
                    gp_ipp_sampling = obs_sampling["gp_ipp"]
                    for i in predict_measurements:
                        observed_value = np.array([0])
                        gp_ipp_sampling.add_observed_point(i, observed_value)
                    gp_ipp_sampling.update_gp()
                    node_info_sampling, node_std_sampling = gp_ipp_sampling.update_node(self.node_coords)

                    obs_sampling["current_position"] = current_coord_sampling
                    obs_sampling["node_info"] = node_info_sampling
                    obs_sampling["node_std"] = node_std_sampling
    
                    # Update inputs for next sampling step
                    node_inputs_sampling, edge_inputs_sampling, budget_inputs_sampling, \
                        current_index_inputs_sampling, pos_embedding_inputs_sampling = \
                        self.construct_input(self.node_coords, node_edges, pos_embedding,
                                            obs_sampling, gaussian_mean, gaussian_cov,
                                            remain_budget_sampling)

                    # Check if done sampling
                    next_edge_sampling = torch.gather(edge_inputs_sampling, 1,
                                                      next_node_index_sampling.repeat(1, 1, self.k_size))
                    next_edge_sampling = next_edge_sampling.permute(0, 2, 1)
                    connected_nodes_budget_sampling = torch.gather(budget_inputs_sampling, 1, next_edge_sampling)
                    connected_nodes_budget_sampling = connected_nodes_budget_sampling.squeeze(0).squeeze(0)
                    connected_nodes_budget_sampling = connected_nodes_budget_sampling.tolist()
                    sampling_done = True
                    for i in connected_nodes_budget_sampling[1:]:
                        if i[0] > 0:
                            sampling_done = False

                    # If done sampling
                    if sampling_done or sampling_step == self.sampling_steps - 1:
                        # figure out the best trajectory with cov trace
                        if new_cov_trace < best_cov_trace:
                            best_cov_trace = new_cov_trace
                            best_path_id = path_id
                        break
            next_coord = planned_path[best_path_id][0]
            best_planned_path = planned_path[best_path_id]
            ## Finish planning

            # Update self intent
            planned_path_intent = [coord for path in planned_path for coord in path]
            mean, cov = calculate_intent_mean_cov(planned_path_intent)
            self.shared_memory["intent_mean"][f"{self.agent_id}"] = mean
            self.shared_memory["intent_std"][f"{self.agent_id}"] = cov

            # Wait for other agents to catch up
            while self.is_agent_ahead(self.agent_id):
                time.sleep(0.05)
            
            # Check done
            budget_remaining_after_step = self.budget_remaining - np.linalg.norm(self.current_position - next_coord)
            check_budget_inputs = self.calc_estimate_budget(budget_remaining_after_step, next_coord, self.agent_id)
            check_budget_inputs = torch.tensor(check_budget_inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, num_nodes, 1)
            next_node_index = first_next_node_index_list[best_path_id]
            next_edge = torch.gather(self.edge_inputs, 1, next_node_index.repeat(1, 1, self.k_size))
            next_edge = next_edge.permute(0, 2, 1)
            connected_nodes_budget = torch.gather(check_budget_inputs, 1, next_edge)
            connected_nodes_budget = connected_nodes_budget.squeeze(0).squeeze(0)
            connected_nodes_budget = connected_nodes_budget.tolist()
            done = True
            for i in connected_nodes_budget[1:]:
                if i[0] > 0:
                    done = False

            route = copy.deepcopy(self.shared_memory["agent_route"][f"{self.agent_id}"])
            route = np.array(route)
            reward, obs = self.env.step_coord(self.agent_id, next_coord, route, done, self.lock)

            # Update agent and shared memory
            self.budget_remaining -= np.linalg.norm(self.current_position - next_coord)
            self.current_position = next_coord
            self.shared_memory["agent_position"][f"{self.agent_id}"] = next_coord
            self.shared_memory["agent_route"][f"{self.agent_id}"].append(next_coord)

            # Get intent from shared memory
            with self.lock:
                gaussian_mean = copy.deepcopy(self.shared_memory["intent_mean"])
                gaussian_cov = copy.deepcopy(self.shared_memory["intent_std"])

            # Plotting
            if self.save_image:
                self.plot(obs, gaussian_mean, gaussian_cov, planned_trajectory=best_planned_path)

            # Collect data
            action_index = first_action_index_list[best_path_id]

            node_inputs_data.append(self.node_inputs.squeeze(0).cpu().numpy().copy()) # append (num_nodes, 5)
            edge_inputs_data.append(self.edge_inputs.squeeze(0).cpu().numpy().copy()) # append (num_nodes, max_num_edges)
            budget_inputs_data.append(self.budget_inputs.squeeze(0).cpu().numpy().copy()) # append (num_nodes, 1)
            current_index_inputs_data.append(self.current_index_inputs.squeeze(0).cpu().numpy().copy()) # append (1, 1)
            lstm_h_inputs_data.append(self.lstm_h_inputs.squeeze(0).cpu().numpy().copy()) # append (1, embedding_size)
            lstm_c_inputs_data.append(self.lstm_c_inputs.squeeze(0).cpu().numpy().copy()) # append (1, embedding_size)
            pos_embedding_inputs_data.append(self.pos_embedding_inputs.squeeze(0).cpu().numpy().copy()) # append (num_nodes, 32)
            actions_data.append(action_index.cpu().numpy().copy()) # append (1,)
            terminated_data.append(done)
            reward_data.append(reward)

            # Update lstm states
            self.lstm_h_inputs = first_lstm_h_list[best_path_id]
            self.lstm_c_inputs = first_lstm_c_list[best_path_id]

            if done:
                self.shared_memory["all_done"][f"{self.agent_id}"] = True
                break

        if self.save_image:
            self.make_gif(obs)

        with self.lock:
            perf_metrics = dict()
            perf_metrics['remain_budget'] = self.budget_remaining / self.budget
            perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
            perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
            perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace[f"{self.agent_id}"]
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.gp_ipp.get_high_info_area())
            perf_metrics['cov_trace'] = self.env.cov_trace[f"{self.agent_id}"]
            perf_metrics['success_rate'] = done # no such thing as success rate in this case
        
        self.episode_data["node_inputs"] = np.array(node_inputs_data)
        self.episode_data["edge_inputs"] = np.array(edge_inputs_data)
        self.episode_data["budget_inputs"] = np.array(budget_inputs_data)
        self.episode_data["current_index_inputs"] = np.array(current_index_inputs_data)
        self.episode_data["lstm_h_inputs"] = np.array(lstm_h_inputs_data)
        self.episode_data["lstm_c_inputs"] = np.array(lstm_c_inputs_data)
        self.episode_data["pos_embedding_inputs"] = np.array(pos_embedding_inputs_data)
        self.episode_data["actions"] = np.array(actions_data)
        self.episode_data["terminated"] = np.array(terminated_data)
        self.episode_data["reward"] = np.array(reward_data)

        return perf_metrics
    
    
    def construct_input(self, node_coords, node_edges, pos_embedding, obs, gaussian_mean, gaussian_cov, budget_remaining):
        num_nodes = len(node_coords) # sample_size+2
        current_position = obs["current_position"] # (2,)

        rel_coords = node_coords - current_position # (num_nodes, 2)
        node_info = obs["node_info"].reshape(num_nodes, 1) # (num_nodes, 1)
        node_std = obs["node_std"].reshape(num_nodes, 1) # (num_nodes, 1)
        node_intent = calculate_intent_info(self.num_agent, gaussian_mean, gaussian_cov, self.agent_id, node_coords) # (num_nodes, 1)
        node_inputs = np.concatenate((rel_coords, node_info, node_std, node_intent), axis=1) # (num_nodes, 5)

        budget_inputs = self.calc_estimate_budget(budget_remaining, current_position, self.agent_id) # (num_nodes, 1)

        current_index = self.env.prm[f"{self.agent_id}"].findNodeIndex(current_position) # int

        # threshold = self.env.threshold
        # agent_inputs = np.array([current_position[0], current_position[1], budget_remaining, threshold]).reshape(1, 4)  # (1, 4)
        
        # Convert to PyTorch tensors and add batch dimension
        node_inputs = torch.tensor(node_inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, num_nodes, 5)
        edge_inputs = torch.tensor(node_edges, dtype=torch.int64).unsqueeze(0).to(self.device) # (1, num_nodes, max_num_edges)
        budget_inputs = torch.tensor(budget_inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, num_nodes, 1)
        current_index_inputs = torch.tensor(current_index, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 1)
        pos_embedding_inputs = torch.tensor(pos_embedding, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, num_nodes, 32)

        return node_inputs, edge_inputs, budget_inputs, current_index_inputs, pos_embedding_inputs

    def is_reset_all(self):
        return all(self.shared_memory["all_reset"].values())
    
    def is_agent_ahead(self, agent_id):
        current_agent_dis = self.cal_agent_distance(agent_id)
        for id in range(self.num_agent):
            if id != agent_id and not self.shared_memory["all_done"][f"{id}"]:
                if current_agent_dis > self.cal_agent_distance(id):
                        return True
        return False
    
    def cal_agent_distance(self, agent_id):
        agent_route = self.shared_memory["agent_route"][f"{agent_id}"]
        distance = 0
        for i, t in enumerate(agent_route):
            if i > 0:
                distance += np.linalg.norm(t - agent_route[i - 1])
        return distance

    def calc_estimate_budget(self, budget, current_pos, agent_ID):
        all_budget = []
        # print(f"current_coord is {current_coord}")
        for i, point_coord in enumerate(self.node_coords):
            dist_current2point = self.env.prm[f"{agent_ID}"].calcDistance(current_pos, point_coord)
            estimate_budget = (budget - dist_current2point) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i + 1, 1)
    
    def plot(self, obs, gaussian_mean, gaussian_cov, planned_trajectory=[]):
        with self.lock: # Lock to see only current state
            all_agent_route = copy.deepcopy(self.shared_memory["agent_route"])
        
        gp_ipp_obs = obs["gp_ipp"]
        cov_trace = obs["cov_trace"]

        agent_route = all_agent_route[f"{self.agent_id}"]
        start_pos = agent_route[0]
        agent_step = len(agent_route) - 1
        budget_remaining = self.budget_remaining
        
        colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
        agent_color = colorlist[self.agent_id]

        fig = gp_ipp_obs.plot(self.env.ground_truth) # Plot Pred Mean, Pred Std, Ground Truth

        # Plot Interesting Area
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_ylim(0, 1)  # BUG hopefully this prevents the lines bug
        ax5.set_xlim(0, 1)
        ax5.set_title('Interesting area')
        high_info_area = gp_ipp_obs.get_high_info_area()
        x = high_info_area[:, 0]
        y = high_info_area[:, 1]
        # ax5.hist2d(x, y, bins=30, vmin=0, vmax=1, edgecolors="face")
        
        # Create a 2D histogram manually
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[0, 1], [0, 1]])
        # Plot using imshow
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax5.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)
        
        ax1 = fig.axes[0] # Pred Mean subplot
        ax1.scatter(start_pos[0], start_pos[1], c="r", marker='*', s=15 ** 2) # Start Position
        for id in range(self.num_agent):
            pointsToDisplay = [point for point in all_agent_route[f"{id}"]]
            x = [item[0] for item in pointsToDisplay]
            y = [item[1] for item in pointsToDisplay]
            min_alpha = 0.25
            max_alpha = 1
            for i in range(len(x) - 1):
                alpha = min_alpha + (max_alpha - min_alpha) * (i / (len(x) - 1))
                ax1.plot(x[i:i + 2], y[i:i + 2], c=colorlist[id], linewidth=4, zorder=5, alpha=alpha)

        # Plot own intent route
        x = [item[0] for item in planned_trajectory]
        y = [item[1] for item in planned_trajectory]
        ax1.plot(x, y, c='white', linewidth=4, zorder=4, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Plot intent of other agents
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title('Intent')
        M = 1000  # sample numbers in gaussian distribution
        gaussian_value = np.zeros((M, M))
        for i in range(self.num_agent):
            if len(gaussian_mean[f"{i}"]) != 0 and i != self.agent_id:
                Gaussian = multivariate_normal(mean=gaussian_mean[f"{i}"], cov=gaussian_cov[f"{i}"])
                X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
                d = np.dstack([X, Y])
                Z = Gaussian.pdf(d).reshape(M, M)
                gaussian_value += Z

        # gaussian_value = gaussian_value / np.max(gaussian_value)
        max_value = np.max(gaussian_value)
        if max_value != 0 and not np.isnan(max_value):
            gaussian_value = gaussian_value / max_value
        else:
            # Handle the case where max_value is 0 or NaN
            gaussian_value = np.zeros_like(gaussian_value)
        X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
        levels = [0.01 * i for i in range(101)]

        ax3.contourf(X, Y, gaussian_value, levels, cmap=cm.jet)
        # ax3.colorbar()

        for i in range(self.num_agent):
            if i != self.agent_id and len(gaussian_mean[f"{i}"]) != 0:
                ax3.scatter(gaussian_mean[f"{i}"][0], gaussian_mean[f"{i}"][1], c=colorlist[i], marker='*',
                            s=15 ** 2)
                # for j in range(len(sampling_end_nodes[f"{i}"])):
                #     ax3.scatter(sampling_end_nodes[f"{i}"][j][0], sampling_end_nodes[f"{i}"][j][1], c=colorlist[i],
                #                 marker='o', s=8 ** 2)
            
        fig.suptitle('Color: {} Cov trace: {:.4g}  remain_budget: {:.4g}'.format(agent_color, cov_trace, budget_remaining))
        fig.savefig(f'{self.episode_gifs_path}/agent{self.agent_id}_step{agent_step}.png', dpi=150)
        self.frame_files.append(f'{self.episode_gifs_path}/agent{self.agent_id}_step{agent_step}.png')

        fig.clf()
        plt.close(fig)

    def make_gif(self, obs, delete_files=False, duration=500):
        gif_name = f'{self.episode_gifs_path}/agent{self.agent_id}_cov_trace_{obs["cov_trace"]}.gif'

        with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
            for frame in self.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

        # Remove files
        if delete_files:
            for filename in self.frame_files[:-1]:
                os.remove(filename)

if __name__ == "__main__":
    start_time = time.time()

    meta_agent_id = 0
    num_agent = 3
    shared_memory = dict() # mutable

    shared_memory["reset"] = False # reset flag of shared memory
    shared_memory["initial_env_obs"] = None
    # self.shared_memory["cov_trace"] = dict() # covariance trace for all agents
    shared_memory["agent_route"] = dict() # route of all agents
    shared_memory["agent_position"] = dict() # position of all agents
    shared_memory["all_reset"] = dict() # reset flag for all agents
    shared_memory["all_done"] = dict() # done flag for all agents

    for agent_id in range(num_agent):
        shared_memory["all_reset"][f"{agent_id}"] = False
        shared_memory["all_done"][f"{agent_id}"] = False

    episode_data = dict()
    perf_metrics = dict()

    workers = []
    worker_threads = []
    workerNames = ["worker_" + str(i) for i in range(num_agent)]

    from classes.env.env import Env
    env = Env(num_agent=3,
              sample_size=200,
              measurement_interval=0.2,
              k_size=20,
              start=None,
              destination=None,
              obstacle=[],
              save_image=False,
              seed=None,
              adaptive_area=True,
              threshold=0.4,
              beta=1.0,
              gaussian_num=[8, 12]
            )

    import threading
    lock = threading.Lock() # Shared lock for all workers

    EMBEDDING_SIZE = 128
    SAMPLING_STEPS = 5
    K_SIZE = 20
    from model.common.AttentionNet import AttentionNet
    model = AttentionNet(5, EMBEDDING_SIZE)
    model_weights = model.state_dict()
    for agent_id in range(num_agent):
        workers.append(RLMAIPPWorker(
                                meta_agent_id=meta_agent_id,
                                agent_id=agent_id,
                                episode_number=0,
                                env=env,
                                shared_memory=shared_memory,
                                lock=lock,
                                num_agent=num_agent,
                                model=model,
                                weights=model_weights,
                                num_paths=8,
                                budget=3.0,
                                steps=256,
                                sampling_steps=SAMPLING_STEPS,
                                k_size=K_SIZE,
                                embedding_size=EMBEDDING_SIZE,
                                eval=False,
                                device="cuda",
                                save_image=True,
                                gifs_path='gifs'
                                )
                                )

    for i, w in enumerate(workers):
        worker_work = lambda: w.work()
        t = threading.Thread(target=worker_work, name=workerNames[i])
        t.start()

        worker_threads.append(t)

    cov_trace = None

    for w in workers:
        while w.perf_metrics == None:
            time.sleep(0.5)
        episode_data[w.agent_id] = w.episode_data
        if cov_trace is None:
            cov_trace = w.perf_metrics['cov_trace']
            perf_metrics = w.perf_metrics
        elif cov_trace > w.perf_metrics['cov_trace']:
            cov_trace = w.perf_metrics['cov_trace']
            perf_metrics = w.perf_metrics

    end_time = time.time()
    # print(episode_data)
    print(perf_metrics)
    print(f"Episode {0} completed in {end_time - start_time} seconds")
