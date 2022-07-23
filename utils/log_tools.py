import os
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def del_all_files_in_dir(path):
    ls = os.listdir(path)
    for file in ls:
        os.remove(os.path.join(path, file))


class TensorboardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_train_data(self, log_datas: dict, step):
        for log_data in log_datas.items():
            self.writer.add_scalar("train_data/" + log_data[0], log_data[1], step)
        self.writer.flush()

    def log_learn_data(self, log_datas: dict, step):
        for log_data in log_datas.items():
            self.writer.add_scalar("learn_data/" + log_data[0], log_data[1], step)
        self.writer.flush()

    def log_eval_data(self, log_datas: dict, step):
        for log_data in log_datas.items():
            self.writer.add_scalar("evaluate_data/" + log_data[0], log_data[1], step)
        self.writer.flush()
