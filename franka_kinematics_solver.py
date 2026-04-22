"""
Franka FR3 Robot FK/IK Solver using MuJoCo and Mink
"""

import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R


class FrankaKinematicsSolver:
    def __init__(self, model_path: str, end_site_name: str = "end"):
        """
        Initialize Franka kinematics solver.

        Args:
            model_path: Path to the MuJoCo XML model file
            end_site_name: Name of the end-effector site in the model
        """
        self.model_path = model_path
        self.end_site_name = end_site_name

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Setup Mink configuration for IK
        self.configuration = mink.Configuration(self.model)

        # Get site ID for forward kinematics
        self.end_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            end_site_name
        )

        if self.end_site_id < 0:
            raise ValueError(f"Site '{end_site_name}' not found in model")

        # Setup IK tasks
        self.end_task = mink.FrameTask(
            frame_name=end_site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.01,
            lm_damping=0.5,
        )

        self.posture_task = mink.PostureTask(model=self.model, cost=1e-8)

        # Initialize posture task with neutral configuration
        mujoco.mj_resetData(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        self.ik_solver = "daqp"
        self.damping = 1e-12

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics.

        Args:
            q: Joint positions [7,] in radians

        Returns:
            xyz: End-effector position [3,] in meters
            quat: End-effector orientation quaternion [4,] as [w, x, y, z]
        """
        # Set joint positions
        self.data.qpos[:7] = q

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get site position
        xyz = self.data.site_xpos[self.end_site_id].copy()

        # Get site orientation (rotation matrix)
        xmat = self.data.site_xmat[self.end_site_id].reshape(3, 3).copy()

        # Convert rotation matrix to quaternion [w, x, y, z]
        rot = R.from_matrix(xmat)
        quat = rot.as_quat()  # Returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        return xyz, quat

    def inverse_kinematics(
        self,
        xyz: np.ndarray,
        quat: np.ndarray,
        q_init: np.ndarray = None,
        max_iterations: int = 100,
        dt: float = 0.01,
        pos_threshold: float =1e-1,
        ori_threshold: float = 1e-1
    ) -> tuple[np.ndarray, bool]:
        """
        Compute inverse kinematics.

        Args:
            xyz: Target end-effector position [3,] in meters
            quat: Target end-effector orientation quaternion [4,] as [w, x, y, z]
            q_init: Initial joint configuration [7,]. If None, uses current configuration
            max_iterations: Maximum number of IK iterations
            dt: Integration time step
            pos_threshold: Convergence tolerance for position error (meters)
            ori_threshold: Convergence tolerance for orientation error (radians)

        Returns:
            q: Joint positions [7,] in radians
            success: Whether IK converged successfully
        """
        # Set initial configuration
        if q_init is not None:
            self.configuration.q[:7] = q_init

        # Convert quaternion [w, x, y, z] to rotation matrix
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])  # Convert to [x, y, z, w]
        rot = R.from_quat(quat_scipy)
        rot_mat = rot.as_matrix()

        # Create SE3 target pose
        target_pose = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rot_mat),
            xyz
        )

        # Set IK task target
        self.end_task.set_target(target_pose)

        # Create tasks dictionary
        tasks = {"eef": self.end_task, "posture": self.posture_task}

        # Solve IK with convergence checking
        for i in range(max_iterations):
            # Compute IK velocity
            vel = mink.solve_ik(
                self.configuration,
                tasks.values(),
                dt,
                self.ik_solver,
                damping=self.damping
            )

            # Integrate velocity
            self.configuration.integrate_inplace(vel, dt)

            # Check convergence using task error
            err = self.end_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

            if pos_achieved and ori_achieved:
                return self.configuration.q[:7].copy(), True

        # Did not converge within max iterations
        return self.configuration.q[:7].copy(), False

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get joint limits from the model.

        Returns:
            q_min: Lower joint limits [7,]
            q_max: Upper joint limits [7,]
        """
        q_min = self.model.jnt_range[:7, 0].copy()
        q_max = self.model.jnt_range[:7, 1].copy()
        return q_min, q_max


