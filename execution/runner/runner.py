class Runner:
    def __init__(self, meta_agent_id, num_agent):
        self.meta_agent_id = meta_agent_id
        self.num_agent = num_agent
        self.shared_memory = dict() # mutable

    def init_shared_memory(self):
        # Initialize shared memory for all worker types
        self.shared_memory["reset"] = False # reset flag of shared memory
        self.shared_memory["initial_env_obs"] = None
        # self.shared_memory["cov_trace"] = dict() # covariance trace for all agents
        self.shared_memory["agent_route"] = dict() # route of all agents
        self.shared_memory["agent_position"] = dict() # position of all agents
        self.shared_memory["all_reset"] = dict() # reset flag for all agents
        self.shared_memory["all_done"] = dict() # done flag for all agents

        for agent_id in range(self.num_agent):
            self.shared_memory["all_reset"][f"{agent_id}"] = False
            self.shared_memory["all_done"][f"{agent_id}"] = False
    
    def job(self, episode_number, cfg):
        episode_data, perf_metrics = self.multi_threaded_job(episode_number, cfg)
        
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return episode_data, perf_metrics, info
    
    def multi_threaded_job(self, episode_number, cfg):
        return NotImplementedError