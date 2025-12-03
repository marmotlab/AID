import omegaconf
import hydra

import ray

class Driver:
    def __init__(self, cfg):
        super().__init__()
        ray.init()
        print(f"Driver {cfg.name} Init")
        self.cfg = cfg
        self.num_gpu = cfg.num_gpu
        self.num_meta_agent = cfg.num_meta_agent
        self.num_episode = cfg.num_episode
        self.runner_class = hydra.utils.get_class(cfg.runner._target_)
        self.ray_runner_class = self.create_RayRunner_class(self.runner_class)

    def create_RayRunner_class(self, base_class):
        @ray.remote(num_cpus=1, num_gpus=self.num_gpu / self.num_meta_agent)
        class RayRunner(base_class):
            def __init__(self, meta_agent_id, num_agent):
                super().__init__(meta_agent_id, num_agent)
        return RayRunner
    
    def run(self):
        return NotImplementedError