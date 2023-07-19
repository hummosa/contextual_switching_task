from collections import defaultdict
import numpy as np
import torch

class Logger():
    def __init__(self, config):
        self.config = config
        self.trial_data = defaultdict(list)
        self.timestep_data = defaultdict(list)
        self.testing_data = defaultdict(list)
        self.experiment_name = f'generative_1d_2gaussians_{self.config.seed}_{self.config.use_oracle}'
        self.folder_name = f'./exports/'
    def put_data(self, transition):
        self.timestep_data.append(transition)

    def log_timestep(self, info):
        self.timestep_data['context_names'].append(info['context_names'])
        self.timestep_data['context_ids'].append(info['context_id'])
        self.timestep_data['trial_i'].append(info['trial_i'])
        self.timestep_data['timestep_i'].append(info['timestep_i'])
        self.timestep_data['correct_action'].append(info['correct_action'])
        self.timestep_data['action'].append(info['action'])
        self.timestep_data['reward'].append(info['reward'])
        self.timestep_data['obs'].append(info['obs'])
        self.timestep_data['predictions'].append(info['predictions'])
    def log_all(self, info):
        for key in info.keys():
            self.timestep_data[key].append(info[key])
    def log_switches(self, info):
        self.timestep_data['context_switches'].append(info['context_switches'])
    def log_tests(self, info):
        for key in info.keys():
            self.testing_data[key].append(info[key])
    def save_data(self):
        import csv
        with open(self.config.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.timestep_data)
            
    def print_data(self):
        for row in self.timestep_data:
            print(row)

