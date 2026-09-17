"""Generic robot adapter used to decouple workspace analysis from a specific robot API."""


class RobotLink:
    """
    Class Method Link for Use with Any Robot and Workspace Analyzer.
    Any Robot used must implement some form of the following functions:
    1) Forward Kinematics
    2) Inverse Kinematics
    3) Retrieval of End Effector Transform
    """

    def __init__(self, robot):
        """Binds Robot Object into Analyzer."""
        self.robot = robot
        self.fk_func = None
        self.ik_func = None
        self.ee_func = None
        self.jacobian_space = None
        self.joint_mins = []
        self.joint_maxs = []
        self.methods_bound = [0, 0, 0, 0]

        #Support for Collision Checking
        self.link_names = None
        self.vis_props = None #Used as a backup if col props unavailable
        self.col_props = None #Populate first
        #Col/Vis Props Format:
        #[[type, origin, properties],[]]
        #[['msh', [0, 0, 0, 0, 0, 0], ['../example_mesh', 1.0]]]
        #Check 'completeGeometryParse in arm_model for more info'

    def is_ready(self):
        """
        Readiness Check for robot link system

        Returns:
            system readiness
        """
        if sum(self.methods_bound) == len(self.methods_bound):
            print('Ready to Proceed')
            return True
        else:
            return False

    def print_unbound(self):
        """Prints any unbound methods"""
        if not self.methods_bound[0]:
            print('FK Method Unbound')
        if not self.methods_bound[1]:
            print('IK Method Unbound')
        if not self.methods_bound[2]:
            print('EE Method Unbound')

    def bind_fk(self, FKMethod):
        """
        Binds a FK function to the helper class, ensuring wide range compatibility.
        Args:
            FKMethod: FK method to bind
        """
        self.fk_func = FKMethod
        self.methods_bound[0] = 1

    def bind_ik(self, IKMethod):
        """
        Binds a IK function to the helper class, ensuring wide range compatibility.
        Args:
            IKMethod: IK method to bind
        """
        self.ik_func = IKMethod
        self.methods_bound[1] = 1

    def bind_ee(self, EEMethod):
        """
        Binds a Get End Effector  function to the helper class, ensuring wide range compatibility.
        Args:
            EEMethod: EE method to bind
        """
        self.ee_func = EEMethod
        self.methods_bound[2] = 1

    def bind_jt(self, JTMethod):
        """
        Binds a Joint Transform Getter function to the helper class,
        ensuring wide range compatibility.
        Args:
            JTMethod: JT method to bind
        """
        self.JTFunc = JTMethod
        self.methods_bound[3] = 1

    def FK(self, thetas):
        """
        Performs FK

        Args:
            thetas: Flattened Numpy FLoat Array Equal to the number of Dofs of the robot.
        Returns:
            EEPos: End Effector Pose (Transformation Matrix)
            Success: Whether or not the kinematic operation was successful
        """
        return self.fk_func(thetas)

    def IK(self, desiredTM):
        """
        Performs IK

        Args:
            desiredTM: 4x4 transformation matrix indicating desired
            position of end effector in global space
        Returns:
            success: Boolean describing if operation was successful
        """
        return self.ik_func(desiredTM)

    def getEE(self):
        """
        Get End Effector Position

        Args:
            None
        Returns:
            End Effector 4x4 Transformation Matrix
        """
        return self.ee_func()

    def getJointTransforms(self):
        """
        Get Joint Transforms

        Args:
            None
        Returns:
            Transformation Matrix of Joints
        """
        return self.JTFunc()

    def jacobianBody(self, theta):
        return self.get_jacobian_body(theta)

    def get_jacobian_body(self, theta):
        """
        Temporary Hack, Dependent on Faser Robotics Lib
        """
        return self.robot.jacobianBody(theta)
