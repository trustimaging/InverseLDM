
from invldm.utils.setup import setup_sampling
from invldm.runners.sampler import Sampler

if __name__ == "__main__":
    # Gather arguments from CL and from YAML config file,
    # adjust and check for arguemnts
    # create necessary folders for sampling
    # Passes sampling_only=True flag to autoencoder,
    # diffusion and data namespaces
    args = setup_sampling()

    # With given args, load pre-trained model and data conditioning
    sampler = Sampler(args)

    # Sample autoencoder (fisrt stage model) and diffusion
    # in sequence
    sampler.sample()
