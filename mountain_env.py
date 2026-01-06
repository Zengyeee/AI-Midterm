# mountain_env.py
import os
import pybullet as p

def make_arena(arena_size=20, wall_height=2.0, wall_thickness=0.5, floor_thickness=0.5, pid=None):
    kwargs = {}
    if pid is not None:
        kwargs["physicsClientId"] = pid

    # Floor
    floor_col = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[arena_size/2, arena_size/2, floor_thickness/2],
        **kwargs
    )
    floor_vis = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[arena_size/2, arena_size/2, floor_thickness/2],
        rgbaColor=[0.95, 0.95, 0.95, 1],
        **kwargs
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_col,
        baseVisualShapeIndex=floor_vis,
        basePosition=[0, 0, -floor_thickness/2],
        **kwargs
    )

    # Walls (+Y / -Y)
    wall_col_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], **kwargs)
    wall_vis_y = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1], **kwargs)
    p.createMultiBody(0, wall_col_y, wall_vis_y, [0,  arena_size/2, wall_height/2], **kwargs)
    p.createMultiBody(0, wall_col_y, wall_vis_y, [0, -arena_size/2, wall_height/2], **kwargs)

    # Walls (+X / -X)
    wall_col_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], **kwargs)
    wall_vis_x = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1], **kwargs)
    p.createMultiBody(0, wall_col_x, wall_vis_x, [ arena_size/2, 0, wall_height/2], **kwargs)
    p.createMultiBody(0, wall_col_x, wall_vis_x, [-arena_size/2, 0, wall_height/2], **kwargs)

def load_mountain(pid=None, shapes_dir="shapes", urdf_name="gaussian_pyramid.urdf",
                  position=(0, 0, -1), euler=(0, 0, 0)):
    kwargs = {}
    if pid is not None:
        kwargs["physicsClientId"] = pid

    shapes_path = os.path.join(os.path.dirname(__file__), shapes_dir)
    p.setAdditionalSearchPath(shapes_path, **kwargs)

    orn = p.getQuaternionFromEuler(euler)
    mid = p.loadURDF(urdf_name, position, orn, useFixedBase=1, **kwargs)
    return mid
