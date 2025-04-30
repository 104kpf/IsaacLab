# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:


* :obj:`LDSC_bipedal`: LDSC Bipedal robot with minimal collision bodies


"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

BIPEDAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_LDSC_bipedal/ldsc_bipedal/assets/Robots/bipedal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_Hip_Yaw": 0.0,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Pitch": 0.0,  
            ".*_Knee_Pitch": 0.0,
            ".*_Ankle_Pitch": 0.0,  
            ".*_Ankle_Roll": 0.0,  
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "effort_actuator": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=30,
            velocity_limit=100.0,
            stiffness={
                ".*_Hip_Yaw": 100.0,
                ".*_Hip_Roll": 100.0,
                ".*_Hip_Pitch": 100.0,  
                ".*_Knee_Pitch": 20.0,
                ".*_Ankle_Pitch": 40.0,  
                ".*_Ankle_Roll": 40.0,  
            },
            damping={
                ".*_Hip_Yaw": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Pitch": 5.0,  
                ".*_Knee_Pitch": 3.0,
                ".*_Ankle_Pitch": 3.0,  
                ".*_Ankle_Roll": 3.0,  
            },
        ),
    },
)