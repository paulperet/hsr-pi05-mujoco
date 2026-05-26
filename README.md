Install:

```bash
git clone 
cd HSR-ENV
python -m venv .venv
source .venv/bin/activate
pip install mujoco
```

Visualize the envs:

```bash
python -m mujoco.viewer --mjcf=hsr/models/world.xml
python -m mujoco.viewer --mjcf=hsr/models/cupboard-world.xml
```