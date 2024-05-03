from torch.utils.tensorboard import SummaryWriter
import shutil, os
from datetime import datetime

class Graph:
    def __init__(self, env_name):
        # Utilisation de datetime.now pour obtenir un timestamp unique pour le dossier de log
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = f'runs/{env_name}/{timestamp}'
        self.create_log_dir()  # Renommage de la méthode
        self.writer = SummaryWriter(self.log_dir)

    def create_log_dir(self):
        # Création du dossier sans suppression préalable des données existantes
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Log directory created: {self.log_dir}")

    def log_metrics(self, average_fitness, best_fitness, generation):
        self.writer.add_scalar('Average Fitness', average_fitness, generation)
        self.writer.add_scalar('Best Fitness', best_fitness, generation)

    def close(self):
        self.writer.close()
