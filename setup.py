import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='basic_robotics',
      version='0.2.14',
      long_description=README,
      long_description_content_type='text/markdown',
      description='Basic Robotics Toolbox Developed in FASER Lab',
      url='https://github.com/64-B1T/basic_robotics',
      author='William Chapin',
      author_email='liam@64b1t.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy <= 1.21.5, >= 1.19',
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
