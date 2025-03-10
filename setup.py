"""Run the Setup for Basic Robotics Packaging."""
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='basic_robotics',
      version='1.0.2',
      long_description=README,
      long_description_content_type='text/markdown',
      description="General Purpose Robotics Package featuring forward and inverse kinematics for serial and parallel robots, static analysis, transformations, path planning, and more.",
      url='https://github.com/64-B1T/basic_robotics',
      author='William Chapin',
      author_email='liam@64b1t.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'numpy',
          'pyserial',
          'scipy',
          'numba',
          'modern_robotics',
          'numpy-stl',
          'descartes',
          'alphashape',
          'trimesh',
          'rtree',
          'requests',
          'sqlalchemy'
      ],
      zip_safe=False)
