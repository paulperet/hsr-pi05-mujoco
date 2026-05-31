import mujoco
import numpy as np

model_path = "hsr/models/world.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_flex_joint")
actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_flex_motor")

d.ctrl[actuator_id] = -1.5
for _ in range(100):
    mujoco.mj_step(m, d)

print(f"arm_flex_joint qpos after 100 steps: {d.qpos[m.jnt_qposadr[joint_id]]}")

for i in range(d.ncon):
    contact = d.contact[i]
    if contact.dist < 0:
        geom1 = m.geom(contact.geom1)
        geom2 = m.geom(contact.geom2)
        name1 = geom1.name if geom1.name else f"mesh:{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, geom1.dataid[0]) if geom1.type==mujoco.mjtGeom.mjGEOM_MESH else geom1.type}"
        name2 = geom2.name if geom2.name else f"mesh:{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, geom2.dataid[0]) if geom2.type==mujoco.mjtGeom.mjGEOM_MESH else geom2.type}"
        print(f"Contact between {name1} and {name2}, dist: {contact.dist}")
