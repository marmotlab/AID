import time
import threading

import omegaconf
import hydra

from .runner import Runner

class RLRunner(Runner):
    def __init__(self, meta_agent_id, num_agent):
        super().__init__(meta_agent_id, num_agent)

    def init_shared_memory(self):
        return super().init_shared_memory()
    
    def job(self, episode_number, cfg, current_weights):
        # return super().job(episode_number, cfg)
        episode_data, perf_metrics = self.multi_threaded_job(episode_number, cfg, current_weights)
        
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return episode_data, perf_metrics, info


    def multi_threaded_job(self, episode_number, cfg, current_weights):
        start_time = time.time()
        self.init_shared_memory()

        episode_data = dict()
        perf_metrics = dict()

        workers = []
        worker_threads = []
        workerNames = ["worker_" + str(i) for i in range(cfg.num_agent)]

        self.env = hydra.utils.instantiate(cfg.env)

        lock = threading.Lock() # Shared lock for all workers

        '''
        BUG FIX:
        OmegaConf does funky stuff with data structures when using instantiate
        Hence, use _target_ for class and create object normally
        '''
        Worker = hydra.utils.get_class(cfg.worker._target_)
        worker_args = {k: v for k, v in cfg.worker.items() if k != "_target_"}

        for agent_id in range(cfg.num_agent):
            workers.append(Worker(**worker_args,
                                  meta_agent_id=self.meta_agent_id,
                                  agent_id=agent_id,
                                  episode_number=episode_number,
                                  env=self.env,
                                  shared_memory=self.shared_memory,
                                  lock=lock,
                                  weights=current_weights))

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
        print(f"Episode {episode_number} completed in {end_time - start_time} seconds")
        return episode_data, perf_metrics
