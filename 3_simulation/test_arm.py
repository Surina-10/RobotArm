# ============================================================================
# APPROACH 3: loading the arm into the Genesis physics simulator
#
# Loads the arm's URDF model (its links, joints and 3D meshes) into a Genesis
# scene with a ground plane and prints how many joints (degrees of freedom) it has.
# ============================================================================

import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=False)

plane = scene.add_entity(gs.morphs.Plane())

arm = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/lss_arm.urdf",
        pos=(0.0, 0.0, 0.0),
        fixed=True
    )
)

scene.build()

print("--------------------")
print("Lynxmotion Arm loaded")
print(f"Number DoFs: {arm.n_dofs}")
print("--------------------")
