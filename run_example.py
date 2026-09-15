import sys
import cProfile
sys.path.append('examples')
from example_3js_vis import run_example 
cProfile.run('run_example()', sort='cumtime')
