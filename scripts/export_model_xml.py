#!/usr/bin/env python3
"""Export MuJoCo model XML from the Imitation environment.

Loads the checkpoint config and creates the Imitation environment
to export the scaled rodent model with skin.
"""

import sys
from pathlib import Path

# Add paths
TRACK_MJX_PATH = Path(__file__).parent.parent.parent / "track-mjx"
VNL_PATH = Path(__file__).parent.parent.parent / "vnl-playground"
sys.path.insert(0, str(TRACK_MJX_PATH))
sys.path.insert(0, str(VNL_PATH))

import mujoco
import numpy as np
from etils import epath
from vnl_playground.tasks.rodent import consts
from vnl_playground.tasks.utils import dm_scale_spec
from track_mjx.agent import checkpointing


def main():
    # Load checkpoint config
    checkpoint_path = str(TRACK_MJX_PATH / "model_checkpoints" / "251224_200417_996678")
    print(f"Loading config from checkpoint: {checkpoint_path}")

    ckpt = checkpointing.load_checkpoint_for_eval(checkpoint_path)
    cfg = ckpt["cfg"]
    rescale_factor = cfg.walker_config.rescale_factor

    print(f"Rescale factor from config: {rescale_factor}")

    # Load arena spec
    arena_xml_path = str(consts.ARENA_XML_PATH)
    arena_spec = mujoco.MjSpec.from_file(arena_xml_path)

    # Load rodent spec - remove skin line first as it references unsuffixed body names
    walker_xml_path = str(consts.RODENT_XML_PATH)
    with open(walker_xml_path, 'r') as f:
        rodent_xml = f.read()

    # Comment out the skin line
    rodent_xml = rodent_xml.replace(
        '<skin name="skin" file="rodent_walker_skin.skn"/>',
        '<!-- <skin name="skin" file="rodent_walker_skin.skn"/> -->'
    )

    rodent_spec = mujoco.MjSpec.from_string(rodent_xml)

    # Convert motors to torque-mode actuators
    print("Converting to torque actuators...")
    for actuator in rodent_spec.actuators:
        if actuator.forcerange.size >= 2:
            actuator.gainprm[0] = actuator.forcerange[1]
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.biasprm = np.zeros((10, 1))

    # Scale the rodent
    print(f"Scaling rodent by factor {rescale_factor}...")
    rodent_spec = dm_scale_spec(rodent_spec, rescale_factor)

    # Attach rodent to arena with suffix (standard approach)
    suffix = "-rodent"
    spawn_site = arena_spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
    spawn_body = spawn_site.attach_body(rodent_spec.body("walker"), "", suffix=suffix)
    spawn_body.add_freejoint(name="root")

    # Export the XML
    xml_string = arena_spec.to_xml()

    # Add back the skin reference in the asset section
    xml_string = xml_string.replace(
        '</asset>',
        '    <skin name="skin" file="rodent_walker_skin.skn"/>\n  </asset>'
    )

    # Create a modified skin file with suffixed body names
    # MuJoCo skin format: bone names are 40-byte fixed-width null-padded strings
    skin_path = VNL_PATH / "vnl_playground" / "tasks" / "rodent" / "xmls" / "rodent_walker_skin.skn"
    with open(skin_path, 'rb') as f:
        skin_data = bytearray(f.read())

    # Get all body names from the model
    body_names = set()
    for body in arena_spec.bodies:
        name = body.name
        # Remove suffix if present (we need the unsuffixed names for the skin)
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        if name and name not in ("floor", "walker", "world"):
            body_names.add(name)

    # Also include known names from the rodent model that might be bones
    bone_names = body_names | {
        "torso", "skull", "jaw", "pelvis",
        "upper_leg_L", "lower_leg_L", "foot_L", "toe_L", "hand_L",
        "upper_leg_R", "lower_leg_R", "foot_R", "toe_R", "hand_R",
        "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
        "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R",
        "vertebra_1", "vertebra_2", "vertebra_3", "vertebra_4", "vertebra_5", "vertebra_6",
        "vertebra_axis", "vertebra_atlant",
    } | {f"vertebra_C{i}" for i in range(1, 32)} | {f"vertebra_cervical_{i}" for i in range(1, 10)}

    # MuJoCo skin files use 40-byte fixed-width bone names
    BONE_NAME_SIZE = 40

    # Find and replace bone names in the skin data
    # Sort by length descending to handle longest matches first
    for bone_name in sorted(bone_names, key=len, reverse=True):
        old_bytes = bone_name.encode('ascii')
        new_name = bone_name + suffix
        new_bytes = new_name.encode('ascii')

        # Search for the bone name as a null-padded fixed-width field
        i = 0
        while i < len(skin_data) - BONE_NAME_SIZE:
            # Check if this position matches the bone name
            if skin_data[i:i+len(old_bytes)] == old_bytes:
                # Check if followed by null bytes (within the 40-byte field)
                if i + len(old_bytes) < len(skin_data) and skin_data[i + len(old_bytes)] == 0:
                    # Verify this looks like a bone name field (has null padding)
                    # Replace in place if new name fits in 40 bytes
                    if len(new_bytes) <= BONE_NAME_SIZE:
                        # Replace the name
                        for j, byte in enumerate(new_bytes):
                            skin_data[i + j] = byte
                        # Null-pad the rest
                        for j in range(len(new_bytes), min(BONE_NAME_SIZE, len(old_bytes) + 1)):
                            if i + j < len(skin_data):
                                skin_data[i + j] = 0
                        i += len(new_bytes)
                        continue
            i += 1

    # Write modified skin file
    output_skin_path = Path(__file__).parent.parent / "public" / "models" / "rodent_walker_skin.skn"
    with open(output_skin_path, 'wb') as f:
        f.write(skin_data)
    print(f"Modified skin file written to: {output_skin_path}")

    # Write to output file
    output_path = Path(__file__).parent.parent / "public" / "models" / "rodent_scaled.xml"
    print(f"Writing XML to: {output_path}")

    with open(output_path, "w") as f:
        f.write(xml_string)

    print("Done!")

    # Verify by compiling (without skin in XML)
    model = arena_spec.compile()
    print(f"\nModel info:")
    print(f"  nq: {model.nq}")
    print(f"  nv: {model.nv}")
    print(f"  nu: {model.nu}")
    print(f"  nbody: {model.nbody}")


if __name__ == "__main__":
    main()
