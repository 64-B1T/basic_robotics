Examples
========

Runnable, self-contained scripts live under `examples/
<https://github.com/64-B1T/basic_robotics/tree/main/examples>`_ in the
repository. Each one exposes a ``run_example()`` function and can also be run
directly, e.g.:

.. code-block:: bash

   python examples/example_tm.py

Transformations
----------------

``example_tm.py`` walks through building ``tm`` transforms from lists,
arrays, rotation matrices, and quaternions, and combining them with the
overloaded arithmetic operators.

.. literalinclude:: ../examples/example_tm.py
   :language: python
   :linenos:

Serial Arm (manual construction)
---------------------------------

``example_arm.py`` builds a 6-DOF serial arm by hand from screw axes, link
transforms, and per-link inertia, then draws it with matplotlib.

.. literalinclude:: ../examples/example_arm.py
   :language: python
   :linenos:

Serial Arm (minimal setup + static forces)
--------------------------------------------

``example_arm_minimal.py`` builds the same arm with only the parameters
required for kinematics, then computes joint torques and cross-moments
needed to resist an applied end-effector wrench.

.. literalinclude:: ../examples/example_arm_minimal.py
   :language: python
   :linenos:

Stewart Platform
-----------------

``example_sp.py`` loads a Stewart platform from a JSON parameter file and
draws it with matplotlib.

.. literalinclude:: ../examples/example_sp.py
   :language: python
   :linenos:

Stewart Platform Static Forces
--------------------------------

``example_sp_forces.py`` extends the basic Stewart platform example with an
applied top-plate wrench, solving for the resulting leg forces.

.. literalinclude:: ../examples/example_sp_forces.py
   :language: python
   :linenos:

Path Planning
--------------

``example_path_planning.py`` loads an arm from a URDF and plans a
collision-aware joint-space path between two configurations with RRT*.

.. literalinclude:: ../examples/example_path_planning.py
   :language: python
   :linenos:

Three.js Web Visualization
----------------------------

``example_3js_vis.py`` streams an arm's pose to the bundled three.js web
client. Start the server first with
``python -m basic_robotics.plotting.vis_3js_server``.

.. literalinclude:: ../examples/example_3js_vis.py
   :language: python
   :linenos:

OPC UA Communications
-----------------------

``example_ua_comm.py`` demonstrates wiring up ``OPCUA_Client`` com ports
against a running OPC UA server (e.g. the Prosys demo server).

.. literalinclude:: ../examples/example_ua_comm.py
   :language: python
   :linenos:
