from basic_robotics.general import tm
from basic_robotics.kinematics import loadArmFromURDF
from basic_robotics.path_planning import RRTStar, PathNode
from basic_robotics.plotting.Draw import *

def run_example():
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(0,2)
    arm = loadArmFromURDF('tests/test_helpers/irb_2400.urdf')

    #Generate an RRT* instance
    init = arm.getEEPos()
    rrt = RRTStar(init)
    rrt.addObstruction([0.5, 0.5, 0.7], [1.2, 1.2, 1.2]) # Add some random obstructions
    rrt.addObstruction([0.5, 0.5, -2], [1, 1, 0.5])
    DrawArm(arm, ax)

    goal = arm.FK(np.array([np.pi/3, np.pi/3, -np.pi/8, np.pi/10, -np.pi/4, np.pi/5]))
    arm.FK(np.zeros(6))

    DrawObstructions(rrt.obstructions, ax) #Draw the obstructions for visulization

    #Find a path through the environment
    random.seed(10)
    traj = rrt.findPathGeneral(
    lambda: rrt.generalGenerateTree( # Generate a general RRT* tree using:
        lambda : PathNode(arm.randomPos()), # Random generation for path nodes
        lambda x, y : rrt.distance(x, y), # Distance between nodes as a cost
        lambda x, y : rrt.armObstruction(arm,x,y)), #Basic rtree collision detection as obstruction checking
    goal) # Goal position

    DrawRRTPath(traj, ax, 'green') # Draw the finalized path
   

    plt.show() #Show the plot

if __name__ == '__main__':
    run_example()