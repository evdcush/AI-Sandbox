import torch
from torch.utils.data import DataLoader
import tqdm

class Trainer:
    """
    NB: Trainer objects do not know about the database.
    """
    def __init__(self, model, opt, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None):
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data  = test_data
        self.batch_size = batch_size
        self.device = None
        self.task_id = None

    def _set_id(self, num):
        self.task_id = num

    def save_checkpoint(self, checkpoint_path):
        checkpoint = dict(
            model=self.model.state_dict(),
            opt=self.opt.state_dict(),
            batch_size=self.batch_size
        )
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.batch_size = checkpoint['batch_size']

    def train(self):
        self.model.train()
        dataloader = tqdm.tqdm(DataLoader(self.train_data, self.batch_size, True),
                               desc=f"Train (task {self.task_id})",
                               ncols=80, leave=True)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            y_hat = self.model(x)
            loss  = self.loss_fn(y_hat, y)
            loss.backward()
            self.opt.step()

    def eval(self):
        self.model.eval()
        dataloader = tqdm.tqdm(DataLoader(self.test_data, self.batch_size, True),
                               desc=f"Eval (task {self.task_id})",
                               ncols=80, leave=True)
        correct = total = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            y_pred = y_hat.argmax(1)
            total += y.size(0)
            correct += y_pred.eq(y).sum().item()
        acc = 100 * correct / total
        return acc



