import os
import ray
import csv
import numpy as np

from .driver import Driver

class EvalDriver(Driver):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        cov_trace_across_runs = []
        rmse_across_runs = []
        time_across_runs = []
        meta_agents = [self.ray_runner_class.remote(i, self.cfg.num_agent) for i in range(self.num_meta_agent)]

        try:
            for run_id in range(self.cfg.num_run):
                print(f"Starting Run {run_id}")
                if self.cfg.save_image:
                    self.cfg.worker.gifs_path = self.cfg.gifs_path + f"/run_{run_id}"
                    os.makedirs(self.cfg.worker.gifs_path, exist_ok=True)

                current_episode = 0
                completed_episodes = 0

                cov_trace_across_episodes = []
                rmse_across_episodes = []
                time_across_episodes = []

                jobList = []
                for i, meta_agent in enumerate(meta_agents):
                    self.cfg.env.seed = current_episode
                    jobList.append(meta_agent.job.remote(current_episode, self.cfg))
                    print(f"MetaAgent {i} starting episode {current_episode}")
                    current_episode += 1
                
                while completed_episodes < self.num_episode:
                    done_id, jobList = ray.wait(jobList, num_returns=1)
                    done_jobs = ray.get(done_id)
                    for job in done_jobs:
                        episode_data, perf_metrics, info = job
                        print(f"MetaAgent {info['id']} finished episode {info['episode_number']}")

                        # Save data
                        completed_episodes += 1

                        cov_trace_across_episodes.append(perf_metrics['cov_trace'])
                        rmse_across_episodes.append(perf_metrics['RMSE'])
                        time_across_episodes.append(perf_metrics['time'])

                        # Add new job
                        if current_episode < self.num_episode:
                            meta_agent = meta_agents[info['id']]
                            self.cfg.env.seed = current_episode
                            jobList.append(meta_agent.job.remote(current_episode, self.cfg))
                            print(f"MetaAgent {info['id']} starting episode {current_episode}")
                            current_episode += 1

                with open(f"{self.cfg.logdir}/cov_trace.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(cov_trace_across_episodes)

                with open(f"{self.cfg.logdir}/rmse.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(rmse_across_episodes)

                with open(f"{self.cfg.logdir}/time.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(time_across_episodes)

                cov_trace_avg = np.array(cov_trace_across_episodes).mean()
                cov_trace_std = np.array(cov_trace_across_episodes).std()
                print(f"Run {run_id}: Average Covariance Trace: {cov_trace_avg}, Std: {cov_trace_std}")

                rmse_avg = np.array(rmse_across_episodes).mean()
                rmse_std = np.array(rmse_across_episodes).std()
                print(f"Run {run_id}: Average RMSE: {rmse_avg}, Std: {rmse_std}")

                time_avg = np.array(time_across_episodes).mean()
                time_std = np.array(time_across_episodes).std()
                print(f"Run {run_id}: Average Time: {time_avg}, Std: {time_std}")

                cov_trace_across_runs.extend(cov_trace_across_episodes)
                rmse_across_runs.extend(rmse_across_episodes)
                time_across_runs.extend(time_across_episodes)

            cov_trace_across_runs_avg = np.array(cov_trace_across_runs).mean(axis=0)
            cov_trace_across_runs_std = np.array(cov_trace_across_runs).std(axis=0)

            rmse_across_runs_avg = np.array(rmse_across_runs).mean(axis=0)
            rmse_across_runs_std = np.array(rmse_across_runs).std(axis=0)
            
            time_across_runs_avg = np.array(time_across_runs).mean(axis=0)
            time_across_runs_std = np.array(time_across_runs).std(axis=0)

            print(f"Average Covariance Trace across all runs: {cov_trace_across_runs_avg}, Std: {cov_trace_across_runs_std}")
            print(f"Average RMSE across all runs: {rmse_across_runs_avg}, Std: {rmse_across_runs_std}")
            print(f"Average Time across all runs: {time_across_runs_avg}, Std: {time_across_runs_std}")

            for a in meta_agents:
                ray.kill(a)

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)
        finally:
            ray.shutdown()
            print("Ray shutdown")
