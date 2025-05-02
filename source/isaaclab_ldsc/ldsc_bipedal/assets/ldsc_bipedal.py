# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:


* :obj:`LDSC_bipedal`: LDSC Bipedal robot with minimal collision bodies


"""

import isaaclab.sim as sim_utils
import os
import sys
# from berkeley_humanoid.actuators import IdentifiedActuatorCfg
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  
sys.path.append(PROJECT_ROOT)

from ldsc_bipedal.actuators import IdentifiedActuatorCfg

from isaaclab.assets.articulation import ArticulationCfg

from ldsc_bipedal.assets import ISAAC_ASSET_DIR


LDSC_BIPEDAL_HIP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_Hip_Yaw", ".*_Hip_Roll"],
    effort_limit=20.0,
    velocity_limit=23,
    saturation_effort=402,
    stiffness={".*": 10.0},
    damping={".*": 1.5},
    armature={".*": 6.9e-5 * 81},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

LDSC_BIPEDAL_HIP_2_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_Hip_Pitch"],
    effort_limit=30.0,
    velocity_limit=20,
    saturation_effort=443,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 9.4e-5 * 81},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

LDSC_BIPEDAL_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_Knee_Pitch"],
    effort_limit=30.0,
    velocity_limit=14,
    saturation_effort=560,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 1.5e-4 * 81},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

LDSC_BIPEDAL_ANKLE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_Ankle_Pitch"],
    effort_limit=20.0,
    velocity_limit=23,
    saturation_effort=402,
    stiffness={".*": 1.0},
    damping={".*": 0.1},
    armature={".*": 6.9e-5 * 81},
    friction_static=1.0,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

LDSC_BIPEDAL_ANKLE_2_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_Ankle_Roll"],
    effort_limit=5.0,
    velocity_limit=42,
    saturation_effort=112,
    stiffness={".*": 1.0},
    damping={".*": 0.1},
    armature={".*": 6.1e-6 * 81},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.005,
)


LDSC_BIPEDAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_ldsc_bipedal/ldsc_bipedal/assets/Robots/bipedal.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.515),
        joint_pos={
            ".*_Hip_Yaw": 0.0,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Pitch": 0.0,  
            ".*_Knee_Pitch": 0.0,
            ".*_Ankle_Pitch": 0.0,  
            ".*_Ankle_Roll": 0.0,  
        },
    ),
    actuators={"hip": LDSC_BIPEDAL_HIP_ACTUATOR_CFG, "hip_2": LDSC_BIPEDAL_HIP_2_ACTUATOR_CFG,
               "knee": LDSC_BIPEDAL_KNEE_ACTUATOR_CFG, "ankle": LDSC_BIPEDAL_ANKLE_ACTUATOR_CFG,
               "ankle_2": LDSC_BIPEDAL_ANKLE_2_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
