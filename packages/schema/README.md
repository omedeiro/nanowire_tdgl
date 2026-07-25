# tdgl3d-schema

Project Object Model (POM) v0 for the tdgl3d platform.

The POM is the single, versioned contract shared by:

- the **studio** frontend (editable object tree),
- the **tdgl3d-server** job service (validation + solver dispatch),
- **AI assistant tools** (structured tool calls mutate POM objects).

All entities — geometry (holes), materials (layers/trilayer), applied fields,
solver settings, simulations, and results — are editable objects inside a
`Project` document, not static files.

## Usage

```python
from tdgl3d_schema import Project, DeviceSpec, SolverSettings

project = Project(name="my-device")
print(project.model_dump_json(indent=2))
```

Export JSON Schema (for TypeScript codegen / validation elsewhere):

```bash
python3 -m tdgl3d_schema.export_json_schema  # writes schema/pom.schema.json
```
