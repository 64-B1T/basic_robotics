import numpy as np
from basic_robotics.general import tm

# Import the disp library to properly view instances of the transformations
from basic_robotics.utilities.disp import disp

#The transformation library allows for seamless usage of rotation matrices and other forms of rotation information encoding.

def run_example():
    identity_matrix = tm()
    disp(identity_matrix, 'identity') #This is just zeros in TAA format.

    #Let's create a few more.
    trans_x_2m = tm([2, 0, 0, 0, 0, 0]) #Translations can be created with a list (Xm, Ym, Zm, Xrad, Yrad, Zrad)
    trans_y_4m = tm(np.array([0, 4, 0, 0, 0, 0])) # Translations can be created with a numpy array
    rot_z_90 = tm([0, 0, 0, 0, 0, np.pi/2]) # Rotations can be declared in radians
    trans_z_2m_neg = tm(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, -2], [0, 0, 0, 1]]))
    # Transformations can be created from rotation matrices

    trans_x_2m_quat = tm([2, 0, 0, 0, 0, 0, 1]) # Transformations can even be declared with quaternions

    list_of_transforms = [trans_x_2m, trans_y_4m, rot_z_90]
    disp(list_of_transforms, 'transform list') # List of transforms will be displayed in columns

    #Operations
    new_simple_transform = trans_x_2m + trans_y_4m #Additon is element-wise on TAA form
    new_transform_multiplied = trans_x_2m @ trans_y_4m #Transformation matrix multiplication uses '@'
    new_double_transform = trans_x_2m * 2 # Multiplication by a scalar is elementwise

    #And more visible in the function list documentation
if __name__ == '__main__':
    run_example()