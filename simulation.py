import os
import pybullet as p
import pybullet_data
from multiprocessing import Pool


class Simulation:
    def __init__(self, sim_id=0, arena_size=20, mountain_urdf="gaussian_pyramid.urdf"):
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

        self.arena_size = arena_size
        self.mountain_urdf = mountain_urdf

        # Make paths robust (Windows-friendly)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.shapes_dir = os.path.join(self.base_dir, "shapes")

        self._mountain_top_z = None
        self._mountain_aabb_min = None
        self._mountain_aabb_max = None

    def _make_arena(self, arena_size=20, wall_height=1):
        """Same idea as cw-envt.py make_arena()."""
        wall_thickness = 0.5

        floor_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
            physicsClientId=self.physicsClientId
        )
        floor_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=self.physicsClientId
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_collision_shape,
            baseVisualShapeIndex=floor_visual_shape,
            basePosition=[0, 0, -wall_thickness],
            physicsClientId=self.physicsClientId
        )

        # Two walls (north/south)
        wall_collision_shape_ns = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
            physicsClientId=self.physicsClientId
        )
        wall_visual_shape_ns = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=self.physicsClientId
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_shape_ns,
            baseVisualShapeIndex=wall_visual_shape_ns,
            basePosition=[0, arena_size / 2, wall_height / 2],
            physicsClientId=self.physicsClientId
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_shape_ns,
            baseVisualShapeIndex=wall_visual_shape_ns,
            basePosition=[0, -arena_size / 2, wall_height / 2],
            physicsClientId=self.physicsClientId
        )

        # Two walls (east/west)
        wall_collision_shape_ew = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
            physicsClientId=self.physicsClientId
        )
        wall_visual_shape_ew = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=self.physicsClientId
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_shape_ew,
            baseVisualShapeIndex=wall_visual_shape_ew,
            basePosition=[arena_size / 2, 0, wall_height / 2],
            physicsClientId=self.physicsClientId
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_shape_ew,
            baseVisualShapeIndex=wall_visual_shape_ew,
            basePosition=[-arena_size / 2, 0, wall_height / 2],
            physicsClientId=self.physicsClientId
        )

    def _build_mountain_env(self):
        """Reset world, build arena + load mountain URDF, compute mountain top height."""
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)

        # Paths like cw-envt.py: pybullet_data + shapes/
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
        if os.path.isdir(self.shapes_dir):
            p.setAdditionalSearchPath(self.shapes_dir, physicsClientId=pid)

        self._make_arena(arena_size=self.arena_size, wall_height=1)

        mountain_position = (0, 0, -1)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))

        mountain_id = p.loadURDF(
            self.mountain_urdf,
            mountain_position,
            mountain_orientation,
            useFixedBase=1,
            physicsClientId=pid
        )

        aabb_min, aabb_max = p.getAABB(mountain_id, physicsClientId=pid)
        self._mountain_aabb_min = aabb_min
        self._mountain_aabb_max = aabb_max
        self._mountain_top_z = aabb_max[2]

        return mountain_id

    def _choose_spawn_pos(self):
        """
        Spawn near the mountain on the ground (not an airdrop).
        We place it just outside the mountain AABB on the -X side.
        """
        # default safe spawn near arena edge
        x = -self.arena_size / 2 + 2.0
        y = 0.0
        z = 0.5

        if self._mountain_aabb_min is not None and self._mountain_aabb_max is not None:
            mn = self._mountain_aabb_min
            mx = self._mountain_aabb_max
            x = mn[0] - 1.0
            y = 0.5 * (mn[1] + mx[1])
            z = 0.5  # slightly above floor so it settles

        # keep inside arena bounds
        x = max(-self.arena_size / 2 + 1.0, min(self.arena_size / 2 - 1.0, x))
        y = max(-self.arena_size / 2 + 1.0, min(self.arena_size / 2 - 1.0, y))
        return (x, y, z)

    def run_creature(self, cr, iterations=2400):
        pid = self.physicsClientId

        # reset tracking each evaluation
        cr.reset_evaluation()

        self._build_mountain_env()

        xml_file = f"temp{self.sim_id}.urdf"
        with open(xml_file, "w") as f:
            f.write(cr.to_xml())

        spawn_pos = self._choose_spawn_pos()

        # Robust URDF load: if it fails, give 0 fitness and move on
        try:
            cid = p.loadURDF(xml_file, spawn_pos, physicsClientId=pid)
        except Exception:
            return
        if cid is None or int(cid) < 0:
            return

        started_recording = False
        base_z = None  # z at first contact (so fitness ~0 at spawn/ground contact)

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)

            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)

            try:
                pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            except Exception:
                # This prevents the run from crashing at gen ~187, etc.
                return

            cr.update_position(pos)

            # Start recording only AFTER first contact (no airdrop cheating)
            if not started_recording:
                contacts = p.getContactPoints(bodyA=cid, physicsClientId=pid)
                if contacts:
                    started_recording = True
                    base_z = float(pos[2])

            if started_recording and base_z is not None:
                z = float(pos[2])

                # anti-flying clamp: don't reward absurd heights
                if self._mountain_top_z is not None:
                    z = min(z, float(self._mountain_top_z) + 0.25)

                height_gain = max(0.0, z - base_z)
                cr.update_max_height(height_gain)

    def update_motors(self, cid, cr):
        try:
            num_joints = p.getNumJoints(cid, physicsClientId=self.physicsClientId)
        except Exception:
            return

        motors = cr.get_motors()
        if len(motors) != num_joints:
            # if something goes weird, don't crash the whole run
            return

        for jid in range(num_joints):
            m = motors[jid]
            try:
                p.setJointMotorControl2(
                    cid,
                    jid,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=m.get_output(),
                    force=5,
                    physicsClientId=self.physicsClientId
                )
            except Exception:
                # ignore single-joint control failures
                continue

    def eval_population(self, pop, iterations):
        for cr in pop.creatures:
            self.run_creature(cr, iterations)


class ThreadedSim:
    # Keep this for completeness, but on Windows you will likely stick to Simulation() only.
    def __init__(self, pool_size):
        self.sims = [Simulation(i) for i in range(pool_size)]

    @staticmethod
    def static_run_creature(sim, cr, iterations):
        sim.run_creature(cr, iterations)
        return cr

    def eval_population(self, pop, iterations):
        pool_args = []
        start_ind = 0
        pool_size = len(self.sims)

        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):
                    break
                sim_ind = i % len(self.sims)
                this_pool_args.append([self.sims[sim_ind], pop.creatures[i], iterations])
            pool_args.append(this_pool_args)
            start_ind += pool_size

        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as ppool:
                creatures = ppool.starmap(ThreadedSim.static_run_creature, pool_argset)
                new_creatures.extend(creatures)

        pop.creatures = new_creatures
