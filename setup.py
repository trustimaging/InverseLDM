from setuptools import setup, find_packages


exclude = ['docs', 'tests', 'examples', 'performance', 'exps']

setup(name='invldm',
      version="0.0.1",
      description="Conditional Latent Diffusion Model for Inverse Problems",
      long_description="""
      Conditional Latent Diffusion Model (Rombach, Robin, et al, 2022) adapted
      for physics-based inverse problems. Here the diffusion model is conditioned with the 
      physical data that drives the inverse problem. The diffusion model then learns
      the mapping between physical data to physical model in a noise-initialised generative
      fashion. """,
      project_urls={
          'Source Code': 'https://github.com/dekape/InverseLDM',
          'Issue Tracker': 'https://github.com/dekape/InverseLDM/issues',
      },
      url='https://github.com/dekape/InverseLDM',
      platforms=["Linux", "Mac OS-X", "Unix"],
      test_suite='pytest',
      author="Deborah Pelacani Cruz",
      author_email='deborah.pelacani-cruz18@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=exclude))