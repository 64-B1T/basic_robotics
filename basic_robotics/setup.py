from setuptools import setup

setup(name='basic_robotics',
      version='0.1',
      description='Basic Robotics Toolbox Developed in FASER Lab',
      url='http://github.com/storborg/funniest',
      author='William Chapin',
      author_email='liam@64b1t.com',
      license='MIT',
      packages=['basic_robotics'],
      install_requires=[
          'numpy',
          'scipy',
          'numba',
          'modern_robotics',
          'numpy-stl',
          'descartes',
          'alphashape',
          'json',
          'trimesh',
      ],
      zip_safe=False)
