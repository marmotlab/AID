import os
import math
import random
import logging

import ray
import wandb
import torch
import pickle
import numpy as np
import hydra
from omegaconf import OmegaConf

from .driver import Driver
from classes.util.timer import Timer

log = logging.getLogger(__name__)

'''
TODO
- use shape_meta to get obs_dim
'''

class RLDriver(Driver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        self.n_agent = cfg.num_agent
        self.num_meta_agent = cfg.num_meta_agent
        self.num_episode_per_train_step = cfg.num_episode_per_train_step

        # Batch size for gradient update
        self.batch_size: int = cfg.train.batch_size

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.lr_decay = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.train.lr_decay_step_size, gamma=cfg.train.lr_decay_gamma)

        # Training params
        self.itr = 0
        self.n_train_itr = cfg.train.n_train_itr
        self.val_freq = cfg.train.val_freq
        # self.n_steps = cfg.train.n_steps
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        ## PPO specific
        # Batch size for logprobs calculations after an iteration --- prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        self.gamma = cfg.train.gamma
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)
        self.update_epochs: int = cfg.train.update_epochs
        self.ppo_clip: float = cfg.train.get("ppo_clip", 0.2)
        self.ent_coef: float = cfg.train.get("ent_coef", 0)
        self.vf_coef: float = cfg.train.get("vf_coef", 0)

        # Gradient accumulation to deal with large GPU RAM usage
        self.accumulated_update_per_epoch = cfg.train.get("accumulated_update_per_epoch", None) # number of times to update per epoch


    def run(self):
        ## Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0

        try:
            print(f"Starting training for {self.n_train_itr} iterations")
            while self.itr < self.n_train_itr:
                print(f"Starting iteration {self.itr}")
                ## Define train or eval (gp_ipp resets env every itr)
                eval_mode = self.itr % self.val_freq == 0
                self.model.eval() if eval_mode else self.model.train()

                ## Data holders
                obs_trajs = {
                    "node_inputs": np.array([]),
                    "edge_inputs": np.array([]),
                    "budget_inputs": np.array([]),
                    "current_index_inputs": np.array([]),
                    "lstm_h_inputs": np.array([]),
                    "lstm_c_inputs": np.array([]),
                    "pos_embedding_inputs": np.array([]),
                }
                action_trajs = np.array([])
                terminated_trajs = np.array([])
                reward_trajs = np.array([])
                # n_agent_steps: number of env steps, not fixed
                # total_steps = num_episode_per_train_step * n_agent * n_agent_steps

                ## Start jobs to collect trajectories from env
                current_weights = self.model.state_dict()
                jobList = []
                episode_lens = [] # list of n_agent_steps (num_episode_per_train_step * n_agent)
                total_cov_trace = 0
                avg_cov_trace = 0

                num_episode = self.num_episode_per_train_step if not eval_mode else self.num_meta_agent
                print(f"{"Train" if not eval_mode else "Eval"}: Starting {num_episode} episodes with {self.num_meta_agent} meta agents")
                current_episode = 0
                completed_episodes = 0

                self.cfg.worker.save_image = True if eval_mode else False
                self.cfg.worker.eval = True if eval_mode else False
                self.cfg.worker.gifs_path = self.cfg.gifs_path + f"/{self.itr}" if eval_mode else self.cfg.gifs_path
                os.makedirs(self.cfg.worker.gifs_path, exist_ok=True)

                ## Init Runners
                meta_agents = [self.ray_runner_class.remote(i, self.cfg.num_agent) for i in range(self.num_meta_agent)]

                for i, meta_agent in enumerate(meta_agents):
                    self.cfg.env.seed = current_episode if eval_mode else None
                    jobList.append(meta_agent.job.remote(current_episode, self.cfg, current_weights))
                    print(f"MetaAgent {i} starting episode {current_episode}")
                    current_episode += 1
        
                while completed_episodes < num_episode:
                    done_id, jobList = ray.wait(jobList)
                    done_jobs = ray.get(done_id)
                    for job in done_jobs:
                        episode_data, perf_metrics, info = job
                        print(f"MetaAgent {info['id']} finished episode {info['episode_number']}")

                        # Save data
                        completed_episodes += 1
                        total_cov_trace += perf_metrics["cov_trace"]

                        for agent_id, agent_episode_data in episode_data.items():
                            episode_len = agent_episode_data["actions"].shape[0] # num of forward passes per episode
                            if action_trajs.size == 0: 
                                for k in obs_trajs:
                                    obs_trajs[k] = agent_episode_data[k]
                                action_trajs = agent_episode_data["actions"]
                                terminated_trajs = agent_episode_data["terminated"] # 1 if last step of episode
                                reward_trajs = agent_episode_data["reward"]
                                print(f"reward: {agent_episode_data["reward"]}")
                            else:
                                for k in obs_trajs:
                                    obs_trajs[k] = np.concatenate((obs_trajs[k], agent_episode_data[k]), axis=0)
                                action_trajs = np.concatenate((action_trajs, agent_episode_data["actions"]), axis=0)
                                terminated_trajs = np.hstack((terminated_trajs, agent_episode_data["terminated"]))
                                reward_trajs = np.hstack((reward_trajs, agent_episode_data["reward"]))
                            cnt_train_step += episode_len if not eval_mode else 0

                            episode_lens.append(episode_len)

                        # Add new job
                        if current_episode < num_episode:
                            meta_agent = meta_agents[info['id']]
                            jobList.append(meta_agent.job.remote(current_episode, self.cfg, current_weights))
                            print(f"MetaAgent {info['id']} starting episode {current_episode}")
                            current_episode += 1

                for a in meta_agents:
                    ray.kill(a)

                total_steps = sum(episode_lens)

                # # Debug holders
                # print(f"Debugging episode outputs")
                # print(f"total_steps {total_steps}")
                # print(f"episode_lens {episode_lens}")
                # print(f"len episode_lens {len(episode_lens)}")
                # print(f"cnt_train_step {cnt_train_step}")
                # for k, v in obs_trajs.items():
                #     print(f"{k}: {v.shape}")
                # print(f"actions {action_trajs.shape}")
                # print(f"terminated {terminated_trajs.shape}")
                # print(f"reward {reward_trajs.shape}")
                # print(f"===============================")

                ## Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
                # purely for logging
                # TODO: Implement success rate and avg best reward
                num_episode_finished = num_episode
                avg_cov_trace = total_cov_trace / num_episode_finished
                split_idx = np.cumsum(episode_lens)[:-1]
                reward_trajs_split = np.split(reward_trajs, split_idx)
                episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
                avg_episode_reward = np.mean(episode_reward)
                success_rate = 1
                avg_best_reward = 1

                ## Update models
                if not eval_mode:
                    with torch.no_grad():
                        ## Prepare data for training
                        for k in obs_trajs:
                            if k == "edge_inputs" or k == "current_index_inputs":
                                obs_trajs[k] = torch.from_numpy(obs_trajs[k]).long().to(self.device)
                            else:
                                obs_trajs[k] = torch.from_numpy(obs_trajs[k]).float().to(self.device)
                        action_trajs = torch.from_numpy(action_trajs).long().to(self.device)
                        # Calculate value and logprobs - split into batches to prevent out of memory
                        num_split = math.ceil(
                            total_steps / self.logprob_batch_size
                        )
                        obs_ts = [{} for _ in range(num_split)]
                        for k in obs_trajs: # for each key in obs_trajs
                            obs_k = obs_trajs[k] # get the np array
                            obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0) # split into batches
                            for i, obs_t in enumerate(obs_ts_k): # for each batch
                                obs_ts[i][k] = obs_t # i is batch index, k is key
                        action_ts = torch.split(action_trajs, self.logprob_batch_size, dim=0) # split into batches

                        values_trajs = np.array([]) # (total_steps)
                        logprobs_trajs = np.array([]) # (total_steps)
                        for obs, actions in zip(obs_ts, action_ts):
                            logp_list, values, _, _ = self.model(*(val for val in obs.values()))
                            logprobs = torch.gather(logp_list, 1, actions).squeeze(-1) # (batch_size, )
                            logprobs = logprobs.cpu().numpy()
                            logprobs_trajs = np.concatenate(
                                (logprobs_trajs, logprobs), axis=0
                            )
                            values = values.squeeze(-1).squeeze(-1).cpu().numpy() # (batch_size, 1, 1) -> (batch_size)
                            values_trajs = np.concatenate(
                                (values_trajs, values), axis=0
                            )

                        ## Free GPU memory
                        del obs_ts, obs_k, obs_ts_k, obs, obs_t
                        del values, logprobs

                        torch.cuda.empty_cache()

                        ## bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                        # If t=T is the last step, V(T+1) = 0
                        advantages_trajs = np.zeros_like(reward_trajs) # (total_steps)

                        # # Debug holders
                        # print(f"Debugging inputs for GAE")
                        # print(f"total_steps {total_steps}")
                        # print(f"episode_lens {episode_lens}")
                        # print(f"cnt_train_step {cnt_train_step}")
                        # for k, v in obs_trajs.items():
                        #     print(f"{k}: {v.shape}")
                        # print(f"actions {action_trajs.shape}")
                        # print(f"terminated {terminated_trajs.shape}")
                        # print(f"reward {reward_trajs.shape}")
                        # print(f"values_trajs {values_trajs.shape}")
                        # print(f"logprobs_trajs {logprobs_trajs.shape}")
                        # print(f"advantages_trajs {advantages_trajs.shape}")
                        # print(f"===============================")

                        for episode_id, episode_len in enumerate(episode_lens): # for each episode
                            lastgaelam = 0
                            offset_index = sum(episode_lens[:episode_id])
                            for t in reversed(range(episode_len)):
                                if t == episode_len - 1:
                                    nextvalues = 0
                                else:
                                    nextvalues = values_trajs[offset_index + t + 1]
                                nonterminal = 1.0 - terminated_trajs[offset_index + t]
                                # delta = r + gamma*V(st+1) - V(st)
                                delta = (
                                    reward_trajs[offset_index + t]
                                    + self.gamma * nextvalues * nonterminal 
                                    - values_trajs[offset_index + t]
                                )
                                # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                                advantages_trajs[offset_index + t] = lastgaelam = (
                                    delta
                                    + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                                )
                            returns_trajs = advantages_trajs + values_trajs

                    ## k for environment step
                    obs_k = obs_trajs
                    actions_k = action_trajs
                    returns_k = torch.from_numpy(returns_trajs).float().to(self.device).reshape(-1)
                    values_k = torch.from_numpy(values_trajs).float().to(self.device).reshape(-1)
                    advantages_k = torch.from_numpy(advantages_trajs).float().to(self.device).reshape(-1)
                    logprobs_k = torch.from_numpy(logprobs_trajs).float().to(self.device)

                    ## Update policy and critic
                    clipfracs = []

                    # # Debug holders
                    # print(f"Debugging inputs for loss")
                    # print(f"total_steps {total_steps}")
                    # print(f"episode_lens {episode_lens}")
                    # print(f"cnt_train_step {cnt_train_step}")
                    # for k, v in obs_k.items():
                    #     print(f"{k}: {v.shape}")
                    # print(f"actions {actions_k.shape}")
                    # print(f"returns {returns_k.shape}")
                    # print(f"values {values_k.shape}")
                    # print(f"advantages {advantages_k.shape}")
                    # print(f"logprobs {logprobs_k.shape}")
                    # print(f"===============================")

                    num_batch = max(1, math.ceil(total_steps / self.batch_size)) # total number of minibatches per epoch
                    accumulation_steps = math.ceil(num_batch / self.accumulated_update_per_epoch) if self.accumulated_update_per_epoch else 1 # number of minibatches to accumulate gradients over

                    for update_epoch in range(self.update_epochs):
                        # for each epoch, go through all data in batches
                        inds_k = torch.randperm(total_steps, device=self.device)
                        update_count = 0
                        for batch in range(num_batch):
                            current_accumulation_steps = accumulation_steps
                            current_accumulation_interval = batch // accumulation_steps # which accumulation interval we are in
                            if self.accumulated_update_per_epoch and current_accumulation_interval == self.accumulated_update_per_epoch - 1: # if last accumulation interval take care of remaining batches
                                current_accumulation_steps = num_batch % accumulation_steps if num_batch % accumulation_steps != 0 else accumulation_steps

                            start = batch * self.batch_size
                            end = min(start + self.batch_size, total_steps)
                            inds_b = inds_k[start:end]  # b for batch
                            obs_b = {k: v[inds_b] for k, v in obs_k.items()}
                            actions_b = actions_k[inds_b]
                            returns_b = returns_k[inds_b]
                            values_b = values_k[inds_b]
                            advantages_b = advantages_k[inds_b]
                            old_logprobs_b = logprobs_k[inds_b]

                            # # Debug holders
                            # print(f"Debugging batched inputs for loss")
                            # print(f"batch_size {inds_b.shape}")
                            # for k, v in obs_b.items():
                            #     print(f"{k}: {v.shape}")
                            # print(f"actions {actions_b.shape}")
                            # print(f"returns {returns_b.shape}")
                            # print(f"values {values_b.shape}")
                            # print(f"advantages {advantages_b.shape}")
                            # print(f"logprobs {old_logprobs_b.shape}")
                            # print(f"===============================")

                            # get loss
                            logp_list, values_pred, _, _ = self.model(*(val for val in obs_b.values()))

                            logprobs = torch.gather(logp_list, 1, actions_b).squeeze(-1) # (batch_size, )
                            log_ratio = logprobs - old_logprobs_b.detach() # (batch_size, )
                            ratio = torch.exp(log_ratio)
                            surr1 = ratio * advantages_b.detach()
                            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages_b.detach()
                            policy_loss = -torch.min(surr1, surr2)
                            policy_loss = policy_loss.mean()


                            value_loss = (returns_b - values_pred.squeeze(-1).squeeze(-1)).pow(2).mean()

                            entropy_loss = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                            # normalize loss for gradient accumulation
                            loss = loss / current_accumulation_steps
                            loss.backward()

                            clipfrac = ((ratio - 1.0).abs() > self.ppo_clip).float().mean().item()
                            approx_kl = ((ratio - 1) - log_ratio).mean().item()
                            ratio = torch.mean(ratio).item()

                            clipfracs += [clipfrac]
                            pg_loss = policy_loss.item()
                            v_loss = value_loss.item()

                            # update policy and critic
                            if (batch + 1) % accumulation_steps == 0 or (batch + 1) == num_batch:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.parameters(),
                                        self.max_grad_norm,
                                    )
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                                update_count += 1
                                  
                                log.info(f"Epoch {update_epoch}\n"
                                         f"Optimizer step {update_count}/{self.accumulated_update_per_epoch}\n"
                                         f"Batch {batch + 1}/{num_batch}\n"
                                         f"approx_kl: {approx_kl}, ratio: {ratio}, clipfrac: {clipfrac}"
                                )

                    # Explained variation of future rewards using value function
                    y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = (
                        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                    )

                    del obs_trajs, obs_k, returns_k, values_k, advantages_k, logprobs_k
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    # Update lr
                    self.lr_decay.step()

                # Save model
                if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                    self.save_model()

                # Log loss and save metrics
                run_results.append(
                    {
                        "itr": self.itr,
                        "step": cnt_train_step,
                    }
                )
                if self.itr % self.log_freq == 0:
                    time = timer()
                    run_results[-1]["time"] = time
                    if eval_mode:
                        log.info(
                            f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg cov trace {avg_cov_trace:8.4f}"
                        )
                        if self.use_wandb:
                            wandb.log(
                                {
                                    "success rate - eval": success_rate,
                                    "avg cov trace - eval": avg_cov_trace,
                                    "avg episode reward - eval": avg_episode_reward,
                                    "avg best reward - eval": avg_best_reward,
                                    "num episode - eval": num_episode_finished,
                                },
                                step=self.itr,
                                commit=False,
                            )
                        run_results[-1]["eval_success_rate"] = success_rate
                        run_results[-1]["eval_episode_reward"] = avg_episode_reward
                        run_results[-1]["eval_best_reward"] = avg_best_reward
                    else:
                        log.info(
                            f"Itr {self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | reward {avg_episode_reward:8.4f} | avg cov trace {avg_cov_trace:8.4f} | t:{time:8.4f}"
                        )
                        if self.use_wandb:
                            wandb.log(
                                {
                                    "total env step": cnt_train_step,
                                    "loss": loss,
                                    "pg loss": pg_loss,
                                    "value loss": v_loss,
                                    "approx kl": approx_kl,
                                    "ratio": ratio,
                                    "clipfrac": np.mean(clipfracs),
                                    "explained variance": explained_var,
                                    "avg cov trace - train": avg_cov_trace,
                                    "avg episode reward - train": avg_episode_reward,
                                    "num episode - train": num_episode_finished,
                                    "lr": self.optimizer.param_groups[0]["lr"],
                                },
                                step=self.itr,
                                commit=True,
                            )
                        run_results[-1]["train_episode_reward"] = avg_episode_reward
                    with open(self.result_path, "wb") as f:
                        pickle.dump(run_results, f)
                self.itr += 1

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)
        finally:
            ray.shutdown()
            print("Ray shutdown")
    
    def save_model(self):
        """
        saves model to disk; no ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
        }  # right now `model` includes weights for `network`, `actor`, `actor_ft`. Weights for `network` is redundant, and we can use `actor` weights as the base policy (earlier denoising steps) and `actor_ft` weights as the fine-tuned policy (later denoising steps) during evaluation.
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")