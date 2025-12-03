import copy

import numpy as np


class Predictor:
    def __init__(self, agent_id, gp_ipp, virtual_measurements, measurement_interval = 0.2):
        self.gp = copy.deepcopy(gp_ipp)
        self.measurement_interval = measurement_interval
        self.agent_id = agent_id
        self.virtual_measurements = virtual_measurements

        for i in virtual_measurements:
            observed_value = np.array([0])
            self.gp.add_observed_point(i, observed_value)

        self.gp.update_gp()

    def predict_step(self, next_coord, current_coord, dist_residual, pred_measurement_coords):
        gp_ipp_pred = copy.deepcopy(self.gp)
        next_coord = np.array(next_coord)
        current_coord = np.array(current_coord)
        dist = np.linalg.norm(current_coord - next_coord)
        remain_length = dist
        next_length = self.measurement_interval - dist_residual

        no_sample = True

        while remain_length > next_length:
            if no_sample:
                sample = (next_coord - current_coord) * next_length / dist + current_coord
                no_sample = False
            else:
                sample = (next_coord - current_coord) * next_length / dist + sample

            pred_measurement_coords.append(sample)
            remain_length -= next_length
            next_length = self.measurement_interval

        for i in pred_measurement_coords:
            observed_value = np.array([0])
            gp_ipp_pred.add_observed_point(i, observed_value)

        gp_ipp_pred.update_gp()
        high_info_area = gp_ipp_pred.get_high_info_area() # if ADAPTIVE_AREA else None
        cov_trace = 0
        if high_info_area.ndim == 1:
            x1 = np.linspace(0, 1, 30)
            x2 = np.linspace(0, 1, 30)
            from itertools import product
            x1x2 = np.array(list(product(x1, x2)))
            y_pred, std = gp_ipp_pred.gp.predict(x1x2, return_std=True)
            debug_info = (f"agent_id: {self.agent_id}\n"
                          f"high_info_area: {high_info_area}\n"
                          f"threshold: {gp_ipp_pred.threshold}\n"
                          f"beta: {gp_ipp_pred.beta}\n"
                          f"y_pred: {y_pred}\n"
                          f"std: {std}")
            
                        # Define the file path
            debug_file_path = f"debug_{self.agent_id}.txt"
            
            # Write the debug information to the file
            with open(debug_file_path, "w") as debug_file:
                debug_file.write(debug_info)

            print("DEBUG INFO WRITTEN TO FILE")
        else:
            cov_trace = gp_ipp_pred.evaluate_cov_trace(high_info_area)

        dist_residual = dist_residual + remain_length if no_sample else remain_length
        return cov_trace, dist, dist_residual, pred_measurement_coords