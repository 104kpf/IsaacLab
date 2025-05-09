# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# import os
# import sys

# CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))  
# sys.path.append(PROJECT_ROOT)

from ldsc_bipedal.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from ldsc_bipedal.assets.ldsc_bipedal import LDSC_BIPEDAL_CFG


@configclass
class LDSCBIPEDALRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = LDSC_BIPEDAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -1.0

        self.observations.policy.height_scan = None
        self.scene.height_scanner = None

@configclass
class LDSCBIPEDALRoughEnvCfg_PLAY(LDSCBIPEDALRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
