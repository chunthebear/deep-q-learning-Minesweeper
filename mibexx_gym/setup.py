from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='mibexx_gym_minesweeper',
    version='0.0.3',
    license='MIT',
    install_requires=['gym', 'numpy'],
    author='Mike Bertram',
    author_email='contact@mibexx.de',
    description='Gym Environment for Minesweeper',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = ['GYM', 'ENVIRONMENT', 'MINESWEEPER', 'REINFORCEMENT LEARNING']
)
