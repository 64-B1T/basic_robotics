import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='basic_robotics',
      version='0.3.20',
      long_description=README,
      long_description_content_type='text/markdown',
      description="General Purpose Robotics Package featuring forward and inverse kinematics for serial and parallel robots, static analysis, transformations, path planning, and more.",
      url='https://github.com/64-B1T/basic_robotics',
      author='William Chapin',
      author_email='liam@64b1t.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy <= 1.22.4, >= 1.21',
          'pyserial',
          'scipy',
          'numba',
          'modern_robotics',
          'numpy-stl',
          'descartes',
          'alphashape',
          'trimesh',
          'rtree',
      ],
      zip_safe=False)
