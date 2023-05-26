
from utils.setup import setup_train
from runners.trainer import Trainer

if __name__ == "__main__":
    args = setup_train()
    trainer = Trainer(args)
    trainer.train()
