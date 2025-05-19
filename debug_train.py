from invldm.utils.setup import setup_train
from invldm.runners.debug_trainer import DebugTrainer

if __name__ == "__main__":
    # Gather arguments from CL and from YAML config file,
    # adjust and check for arguemnts
    # create necessary folders for experiment
    args = setup_train()

    # With given args, set up the debug training model with
    # datasets, dataloaders, model and optimiser instan
    # ciations, and data conditioning
    trainer = DebugTrainer(args)

    # Train autoencoder (if n_epochs > 0) and diffusion
    # with logging of intermediate values
    trainer.train() 