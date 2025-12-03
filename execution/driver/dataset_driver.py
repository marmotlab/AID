import os
import ray
import glob
import numpy as np

from .driver import Driver

class DatasetDriver(Driver):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        dataset = None

        meta_agents = [self.ray_runner_class.remote(i, self.cfg.num_agent) for i in range(self.num_meta_agent)]

        current_episode = 0
        completed_episodes = 0

        cov_trace_total = 0

        dataset_path = os.path.join(self.cfg.logdir)

        jobList = []
        for i, meta_agent in enumerate(meta_agents):
            jobList.append(meta_agent.job.remote(current_episode, self.cfg))
            print(f"MetaAgent {i} starting episode {current_episode}")
            current_episode += 1
        
        try:
            while completed_episodes < self.num_episode:
                done_id, jobList = ray.wait(jobList, num_returns=1)
                done_jobs = ray.get(done_id)
                for job in done_jobs:
                    episode_data, perf_metrics, info = job
                    print(f"MetaAgent {info['id']} finished episode {info['episode_number']}")

                    # Save data
                    completed_episodes += 1
                    cov_trace_total += perf_metrics['cov_trace']

                    for agent_id, agent_episode_data in enumerate(episode_data.values()):
                        if dataset is None:
                            dataset = {key: [] for key in agent_episode_data.keys()}
                            dataset["traj_lengths"] = []
                        for key, value in agent_episode_data.items():
                            dataset[key].append(value)
                        dataset["traj_lengths"].append(len(agent_episode_data["actions"]))

                    # Add new job
                    if current_episode < self.num_episode:
                        meta_agent = meta_agents[info['id']]
                        jobList.append(meta_agent.job.remote(current_episode, self.cfg))
                        print(f"MetaAgent {info['id']} starting episode {current_episode}")
                        current_episode += 1

            print(f"All {self.num_episode} episodes completed")
            print(f"Average Covariance Trace: {cov_trace_total / self.num_episode}")

            for key in dataset.keys():
                if key != "traj_lengths":
                    dataset[key] = np.concatenate(dataset[key], axis=0)
                elif key == "traj_lengths":
                    dataset[key] = np.array(dataset[key])
            
            npz_filename = os.path.join(dataset_path, f"{self.cfg.name}.npz")
            np.savez_compressed(npz_filename, **dataset)
            print(f"Saved all episodes into {npz_filename}")

            for a in meta_agents:
                ray.kill(a)

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)
        finally:
            ray.shutdown()
            print("Ray shutdown")