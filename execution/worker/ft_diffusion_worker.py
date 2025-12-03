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


class FinetuneDiffusionWorker:
    def __init__(self,
                 meta_agent_id, agent_id, episode_number, env, shared_memory, lock, num_agent,
                 diffusion_model, weights, cond_steps, num_paths, action_steps, budget,
                 steps, data_type, normalization_path=None, step_size=0.1, eval=False, device='cpu', save_image=False, gifs_path='gifs'):
        self.meta_agent_id = meta_agent_id
        self.agent_id = agent_id
        self.episode_number = episode_number
        self.env = env
        self.shared_memory = shared_memory
        self.lock = lock
        self.num_agent = num_agent
        self.steps = steps
        self.data_type = data_type
        self.normalization_path = normalization_path
        self.step_size = step_size
        self.device = device

        if self.normalization_path is not None:
            normalization_file = np.load(self.normalization_path, allow_pickle=True)
            actions_normalization = normalization_file["actions"].item()
            self.action_min = actions_normalization["min"]
            self.action_max = actions_normalization["max"]

        # For plotting
        self.save_image = save_image
        self.gifs_path = gifs_path
        self.frame_files = []

        self.episode_gifs_path = '{}/{}'.format(self.gifs_path, self.episode_number)
        if not os.path.exists(self.episode_gifs_path) and self.save_image:
            os.makedirs(self.episode_gifs_path)

        # For Agent
        self.diffusion_model = hydra.utils.instantiate(diffusion_model)
        self.diffusion_model.load_state_dict(weights)
        self.diffusion_model.to(self.device)
        self.eval_mode = eval
        if eval:
            self.diffusion_model.eval()
        self.cond_steps = cond_steps
        self.num_paths = num_paths
        self.action_steps = action_steps

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
        pos_embedding_data = []
        agent_inputs_data = []
        chains_data =[]
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
            node_edges.append(edges)
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

        # Init queues
        self.node_inputs_queue = deque(maxlen=self.cond_steps)
        self.pos_embedding_queue = deque(maxlen=self.cond_steps)
        self.agent_inputs_queue = deque(maxlen=self.cond_steps)

        for step in range(self.steps):
            # Update queues and construct cond
            self.update_queue(self.node_coords,
                              pos_embedding,
                              obs,
                              gaussian_mean,
                              gaussian_cov,
                              self.budget_remaining
                              )
            cond = self.construct_cond()
            
            # Get planned path and chains from model
            with torch.no_grad():
                planned_path, chains = self.diffusion_model(cond=cond,
                                                            deterministic=self.eval_mode,
                                                            return_chain=True,
                                                            )
                chains = chains.cpu().numpy() # (num_paths, denoising_steps, horizon, act)??
                planned_path = planned_path.cpu().detach().numpy() # (num_paths, horizon, 2 (1 if theta))

            # Unnormalize planned path if necessary
            if self.normalization_path is not None:
                planned_path = planned_path * (self.action_max - self.action_min) + self.action_min

            # Convert planned path to absolute coordinates
            if self.data_type == "abs":
                pass
            elif self.data_type == "delta":
                planned_path = np.cumsum(planned_path, axis=1) + self.current_position
            elif self.data_type == "theta":
                x = np.cos(planned_path) * self.step_size
                y = np.sin(planned_path) * self.step_size
                planned_path = np.concatenate([x, y], axis=-1)
                planned_path = np.cumsum(planned_path, axis=1) + self.current_position
            else:
                raise ValueError("Invalid data type")

            # Update self intent
            mean, cov = calculate_intent_mean_cov(planned_path.reshape(-1, 2))
            self.shared_memory["intent_mean"][f"{self.agent_id}"] = mean
            self.shared_memory["intent_std"][f"{self.agent_id}"] = cov

            # Init path predictor
            predictor = Predictor(self.agent_id, obs["gp_ipp"], [], self.env.measurement_interval)

            best_cov_trace = float("infinity")
            new_cov_trace = float("infinity")
            best_path_id = None

            # Loop through all possible trajectories
            for path_id, path in enumerate(planned_path):
                dist_residual_prediction = 0
                predict_measurements = []
                current_coord = self.current_position
                predictor_copy = copy.deepcopy(predictor)
                for next_coord in path:
                    new_cov_trace, dist, dist_residual_prediction, predict_measurements = \
                        predictor_copy.predict_step(next_coord,
                                                    current_coord,
                                                    dist_residual_prediction,
                                                    predict_measurements)
                    current_coord = next_coord
                if new_cov_trace < best_cov_trace:
                    best_cov_trace = new_cov_trace
                    best_path_id = path_id

            best_planned_path = planned_path[best_path_id] # (horizon, 2)
            best_chain = chains[best_path_id] # (denoising_steps, horizon, act)

            executed_path = best_planned_path[:self.action_steps]
            
            accumulated_reward = 0
            done = False
            for next_coord in executed_path:
                # Wait for other agents to catch up
                while self.is_agent_ahead(self.agent_id):
                    time.sleep(0.05)

                # Check done
                dist_to_next = np.linalg.norm(self.current_position - next_coord)
                if dist_to_next > self.budget_remaining:
                    next_direction = (next_coord - self.current_position) / dist_to_next
                    next_coord = self.current_position + next_direction * self.budget_remaining
                    done = True

                route = copy.deepcopy(self.shared_memory["agent_route"][f"{self.agent_id}"])
                route = np.array(route)
                reward, obs = self.env.step_coord(self.agent_id, next_coord, route, done, self.lock)

                accumulated_reward += reward

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

                if done:
                    break
            
            # Collect Data
            node_inputs_cond = np.array(self.node_inputs_queue) # (cond_steps, num_nodes, 5)
            pos_embedding_cond = np.array(self.pos_embedding_queue) # (cond_steps, num_nodes, 32)
            agent_inputs_cond = np.array(self.agent_inputs_queue) # (cond_steps, 4)

            node_inputs_data.append(node_inputs_cond)
            pos_embedding_data.append(pos_embedding_cond)
            agent_inputs_data.append(agent_inputs_cond)
            chains_data.append(best_chain) # (denoising_steps+1, horizon, act)
            terminated_data.append(done)
            reward_data.append(accumulated_reward)
            
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
        self.episode_data["pos_embedding"] = np.array(pos_embedding_data)
        self.episode_data["agent_inputs"] = np.array(agent_inputs_data)
        self.episode_data["chains"] = np.array(chains_data)
        self.episode_data["terminated"] = np.array(terminated_data)
        self.episode_data["reward"] = np.array(reward_data)

        return perf_metrics
    
    def update_queue(self, node_coords, pos_embedding, obs, gaussian_mean, gaussian_cov, budget_remaining):
        num_nodes = len(node_coords) # sample_size+2
        rel_coords = node_coords - obs["current_position"] # (num_nodes, 2)
        node_info = obs["node_info"].reshape(num_nodes, 1) # (num_nodes, 1)
        node_std = obs["node_std"].reshape(num_nodes, 1) # (num_nodes, 1)
        node_intent = calculate_intent_info(self.num_agent, gaussian_mean, gaussian_cov, self.agent_id, node_coords) # (num_nodes, 1)
        node_inputs = np.concatenate((rel_coords, node_info, node_std, node_intent), axis=1) # (num_nodes, 5)

        current_position = obs["current_position"] # (2,)
        threshold = self.env.threshold
        agent_inputs = np.array([current_position[0], current_position[1], budget_remaining, threshold]).reshape(1, 4)  # (1, 4)
        
        # Add to queue, horizon dimension (cond_steps)
        self.node_inputs_queue.append(node_inputs) # (cond_steps, num_nodes, 5)
        self.pos_embedding_queue.append(pos_embedding) # (cond_steps, num_nodes, 32)
        self.agent_inputs_queue.append(agent_inputs) # (cond_steps, 4)

        # Fill the queue if empty
        if len(self.node_inputs_queue) < self.cond_steps:
            self.node_inputs_queue.extend([node_inputs] * (self.cond_steps - len(self.node_inputs_queue)))
            self.pos_embedding_queue.extend([pos_embedding] * (self.cond_steps - len(self.pos_embedding_queue)))
            self.agent_inputs_queue.extend([agent_inputs] * (self.cond_steps - len(self.agent_inputs_queue)))

    def construct_cond(self):
        # Convert deque to numpy array and add batch dimension
        node_inputs_cond = torch.tensor(np.array(self.node_inputs_queue), dtype=torch.float32).unsqueeze(0).to(self.device) # (1, cond_steps, num_nodes, 5)
        pos_embedding_cond = torch.tensor(np.array(self.pos_embedding_queue), dtype=torch.float32).unsqueeze(0).to(self.device) # (1, cond_steps, num_nodes, 32)
        agent_inputs_cond = torch.tensor(np.array(self.agent_inputs_queue), dtype=torch.float32).unsqueeze(0).to(self.device) # (1, cond_steps, 1, 4)

        # Repeat for num_paths
        num_paths = self.num_paths
        node_inputs_cond = node_inputs_cond.repeat(num_paths, 1, 1, 1) # (num_paths, cond_steps, num_nodes, 5)
        pos_embedding_cond = pos_embedding_cond.repeat(num_paths, 1, 1, 1) # (num_paths, cond_steps, num_nodes, 32)
        agent_inputs_cond = agent_inputs_cond.repeat(num_paths, 1, 1, 1) # (num_paths, cond_steps, 1, 4)

        # Convert to PyTorch tensors and move to device
        cond = dict()
        cond["node_inputs"] = node_inputs_cond # (num_paths, cond_steps, num_nodes, 5)
        cond["pos_embedding"] = pos_embedding_cond # (num_paths, cond_steps, num_nodes, 32)
        cond["agent_inputs"] = agent_inputs_cond # (num_paths, cond_steps, 4)
        return cond

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