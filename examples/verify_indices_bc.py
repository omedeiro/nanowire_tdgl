"""Systematic verification of indices, boundary conditions, and applied field.

Compares the Python vectorised index construction against the MATLAB triple-loop
logic (re-implemented here in Python for cross-validation).
"""

import numpy as np
import sys
sys.path.insert(0, "src")

from tdgl3d.core.parameters import SimulationParameters
from tdgl3d.mesh.indices import construct_indices, _linear_index
from tdgl3d.physics.applied_field import AppliedField, build_boundary_field_vectors
from tdgl3d.physics.rhs import BoundaryVectors


def matlab_indices(Nx, Ny, Nz):
    """Re-implement the MATLAB contruct_indices.m triple loop in Python.

    Returns 1-based indices (like MATLAB) then converts to 0-based.
    """
    mj = Nx + 1
    mk = (Nx + 1) * (Ny + 1)
    is_3d = Nz > 1

    M2 = []
    M2B = []

    # Face indices: "all" variants (every j,k pair for x; etc.)
    m1x, m2x, mNx, mNxp1 = [], [], [], []  # i=0, i=1, i=Nx-1, i=Nx
    m1y, m2y, mNy, mNyp1 = [], [], [], []
    m1z, m2z, mNz, mNzp1 = [], [], [], []

    # "inner" variants (restricted to interior j,k)
    m1x_int, m2x_int, mNx_int, mNxp1_int = [], [], [], []
    m1y_int, m2y_int, mNy_int, mNyp1_int = [], [], [], []
    m1z_int, m2z_int, mNz_int, mNzp1_int = [], [], [], []

    klim = Nz + 1 if is_3d else 1

    # MATLAB loops: k=1..klim, j=1..Ny+1, i=1..Nx+1  (1-based)
    for k_1 in range(1, klim + 1):
        for j_1 in range(1, Ny + 2):
            for i_1 in range(1, Nx + 2):
                # Linear index (MATLAB 1-based)
                lin_1 = i_1 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1)

                # --- M2 (interior) ---
                if is_3d:
                    if (k_1 != 1 and k_1 != Nz + 1 and
                        j_1 != 1 and j_1 != Ny + 1 and
                        i_1 != 1 and i_1 != Nx + 1):
                        M2.append(lin_1)
                else:
                    if (j_1 != 1 and j_1 != Ny + 1 and
                        i_1 != 1 and i_1 != Nx + 1):
                        M2.append(lin_1)

                # --- x-face indices (when i==1) ---
                if i_1 == 1:
                    m1x.append(1 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1))
                    m2x.append(2 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1))
                    mNx.append(Ny + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1))
                    mNxp1.append(Ny + 1 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1))

                    if is_3d:
                        cond_int = (k_1 != 1 and k_1 != Nz + 1 and
                                    j_1 != 1 and j_1 != Nx + 1)
                    else:
                        cond_int = (j_1 != 1 and j_1 != Nx + 1)

                    if cond_int:
                        m1x_int.append(m1x[-1])
                        m2x_int.append(m2x[-1])
                        mNx_int.append(mNx[-1])
                        mNxp1_int.append(mNxp1[-1])

                # --- y-face indices (when j==1) ---
                if j_1 == 1:
                    m1y.append(i_1 + (Nx + 1) * (Ny + 1) * (k_1 - 1))
                    m2y.append(i_1 + (Nx + 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1))
                    mNy.append(i_1 + (Nx + 1) * (Ny - 1) + (Nx + 1) * (Ny + 1) * (k_1 - 1))
                    mNyp1.append(i_1 + (Nx + 1) * Ny + (Nx + 1) * (Ny + 1) * (k_1 - 1))

                    if is_3d:
                        cond_int = (k_1 != 1 and k_1 != Nz + 1 and
                                    i_1 != 1 and i_1 != Ny + 1)
                    else:
                        cond_int = (i_1 != 1 and i_1 != Ny + 1)

                    if cond_int:
                        m1y_int.append(m1y[-1])
                        m2y_int.append(m2y[-1])
                        mNy_int.append(mNy[-1])
                        mNyp1_int.append(mNyp1[-1])

                # --- z-face indices (when k==1) ---
                if k_1 == 1:
                    m1z.append(i_1 + (Nx + 1) * (j_1 - 1))
                    m2z.append(i_1 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1))
                    mNz.append(i_1 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * (Nz - 1))
                    mNzp1.append(i_1 + (Nx + 1) * (j_1 - 1) + (Nx + 1) * (Ny + 1) * Nz)

                    if (j_1 != 1 and j_1 != Ny + 1 and
                        i_1 != 1 and i_1 != Nx + 1):
                        m1z_int.append(m1z[-1])
                        m2z_int.append(m2z[-1])
                        mNz_int.append(mNz[-1])
                        mNzp1_int.append(mNzp1[-1])

    # M2B (B-field interior, in interior numbering, 1-based)
    for k_1 in range(1, Nz + 1):
        for j_1 in range(1, Ny + 1):
            for i_1 in range(1, Nx + 1):
                if is_3d:
                    cond = k_1 < Nz - 1 and j_1 < Ny - 1 and i_1 < Nx - 1
                else:
                    cond = j_1 < Ny - 1 and i_1 < Nx - 1
                if cond:
                    M2B.append(i_1 + (Nx - 1) * (j_1 - 1) + (Nx - 1) * (Ny - 1) * (k_1 - 1))

    # Convert MATLAB 1-based to Python 0-based
    M2 = np.array(M2, dtype=np.intp) - 1
    M2B = np.array(M2B, dtype=np.intp) - 1

    m1x = np.array(m1x, dtype=np.intp) - 1
    m2x = np.array(m2x, dtype=np.intp) - 1
    mNx = np.array(mNx, dtype=np.intp) - 1
    mNxp1 = np.array(mNxp1, dtype=np.intp) - 1
    m1x_int = np.array(m1x_int, dtype=np.intp) - 1
    m2x_int = np.array(m2x_int, dtype=np.intp) - 1
    mNx_int = np.array(mNx_int, dtype=np.intp) - 1
    mNxp1_int = np.array(mNxp1_int, dtype=np.intp) - 1

    m1y = np.array(m1y, dtype=np.intp) - 1
    m2y = np.array(m2y, dtype=np.intp) - 1
    mNy = np.array(mNy, dtype=np.intp) - 1
    mNyp1 = np.array(mNyp1, dtype=np.intp) - 1
    m1y_int = np.array(m1y_int, dtype=np.intp) - 1
    m2y_int = np.array(m2y_int, dtype=np.intp) - 1
    mNy_int = np.array(mNy_int, dtype=np.intp) - 1
    mNyp1_int = np.array(mNyp1_int, dtype=np.intp) - 1

    m1z = np.array(m1z, dtype=np.intp) - 1
    m2z = np.array(m2z, dtype=np.intp) - 1
    mNz = np.array(mNz, dtype=np.intp) - 1
    mNzp1 = np.array(mNzp1, dtype=np.intp) - 1
    m1z_int = np.array(m1z_int, dtype=np.intp) - 1
    m2z_int = np.array(m2z_int, dtype=np.intp) - 1
    mNz_int = np.array(mNz_int, dtype=np.intp) - 1
    mNzp1_int = np.array(mNzp1_int, dtype=np.intp) - 1

    # mNx_mask (normal BC mask) — in MATLAB this is mNx from the "all" loop,
    # which includes i=Nx-1 for ALL j,k. Actually in MATLAB:
    #   y1(p.mNx) = 0 — this zeros at i=Nx-1 (0-based) for all j,k
    # Wait — MATLAB's mNx corresponds to i_1=1 iteration where
    #   mNx = Ny + ... which is i = Nx-1 (in 0-based, since MATLAB Ny = Nx
    #   when Nx=Ny... Let me check carefully.

    return {
        "M2": M2, "M2B": M2B,
        "m1x": m1x, "m2x": m2x, "mNx": mNx, "mNxp1": mNxp1,
        "m1x_int": m1x_int, "m2x_int": m2x_int,
        "mNx_int": mNx_int, "mNxp1_int": mNxp1_int,
        "m1y": m1y, "m2y": m2y, "mNy": mNy, "mNyp1": mNyp1,
        "m1y_int": m1y_int, "m2y_int": m2y_int,
        "mNy_int": mNy_int, "mNyp1_int": mNyp1_int,
        "m1z": m1z, "m2z": m2z, "mNz": mNz, "mNzp1": mNzp1,
        "m1z_int": m1z_int, "m2z_int": m2z_int,
        "mNz_int": mNz_int, "mNzp1_int": mNzp1_int,
    }


def verify_indices(Nx, Ny, Nz):
    """Compare Python vectorised indices against MATLAB loop reference."""
    print(f"\n{'='*60}")
    print(f"Verifying indices for Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print(f"{'='*60}")

    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz)
    idx = construct_indices(params)
    ref = matlab_indices(Nx, Ny, Nz)

    all_ok = True

    # interior_to_full vs M2
    match = np.array_equal(np.sort(idx.interior_to_full), np.sort(ref["M2"]))
    print(f"  interior_to_full (M2): len={len(idx.interior_to_full)} vs {len(ref['M2'])}  "
          f"{'✓ MATCH' if match else '✗ MISMATCH'}")
    if not match:
        all_ok = False
        diff = set(idx.interior_to_full) ^ set(ref["M2"])
        print(f"    Diff indices: {sorted(diff)[:10]}...")

    # bfield_interior vs M2B
    match = np.array_equal(np.sort(idx.bfield_interior), np.sort(ref["M2B"]))
    print(f"  bfield_interior (M2B): len={len(idx.bfield_interior)} vs {len(ref['M2B'])}  "
          f"{'✓ MATCH' if match else '✗ MISMATCH'}")
    if not match:
        all_ok = False
        print(f"    Python first 5: {idx.bfield_interior[:5]}")
        print(f"    MATLAB first 5: {ref['M2B'][:5]}")

    # Face indices mapping:
    # Python name        → MATLAB name
    # x_face_lo          → m1x      (i=0)
    # x_first            → m2x      (i=1)
    # x_last             → mNx      (i=Nx-1)
    # x_face_hi          → mNxp1    (i=Nx)
    # x_face_lo_inner    → m1x_int
    # x_first_inner      → m2x_int
    # x_last_inner       → mNx_int
    # x_face_hi_inner    → mNxp1_int

    checks = [
        ("x_face_lo", "m1x"), ("x_first", "m2x"),
        ("x_last", "mNx"), ("x_face_hi", "mNxp1"),
        ("x_face_lo_inner", "m1x_int"), ("x_first_inner", "m2x_int"),
        ("x_last_inner", "mNx_int"), ("x_face_hi_inner", "mNxp1_int"),
        ("y_face_lo", "m1y"), ("y_first", "m2y"),
        ("y_last", "mNy"), ("y_face_hi", "mNyp1"),
        ("y_face_lo_inner", "m1y_int"), ("y_first_inner", "m2y_int"),
        ("y_last_inner", "mNy_int"), ("y_face_hi_inner", "mNyp1_int"),
        ("z_face_lo", "m1z"), ("z_first", "m2z"),
        ("z_last", "mNz"), ("z_face_hi", "mNzp1"),
        ("z_face_lo_inner", "m1z_int"), ("z_first_inner", "m2z_int"),
        ("z_last_inner", "mNz_int"), ("z_face_hi_inner", "mNzp1_int"),
    ]

    for py_name, ml_name in checks:
        py_arr = np.sort(getattr(idx, py_name))
        ml_arr = np.sort(ref[ml_name])
        match = np.array_equal(py_arr, ml_arr)
        status = "✓" if match else "✗"
        if not match:
            all_ok = False
            print(f"  {status} {py_name} ({ml_name}): len {len(py_arr)} vs {len(ml_arr)}")
            if len(py_arr) > 0 and len(ml_arr) > 0:
                print(f"      Python: {py_arr[:8]}")
                print(f"      MATLAB: {ml_arr[:8]}")
        else:
            print(f"  {status} {py_name} ({ml_name}): {len(py_arr)} entries match")

    # --- Normal BC mask verification ---
    # MATLAB: y1(p.mNx) = 0 — p.mNx is the "all" face at i=Nx-1 (0-based)
    # which is our x_last (all j,k pairs at i=Nx-1)
    # But wait — the Python x_normal_bc_mask uses vals=[0, Nx-1].
    # MATLAB mNx: for every (j,k), at i=1 in MATLAB → mNx(h) = Ny + mj*(j-1) + mk*(k-1)
    # = (Nx-1) + mj*(j-1) + mk*(k-1)  in 0-based = i=Nx-1
    # So Python x_normal_bc_mask should be faces at i=0 and i=Nx-1.
    # But MATLAB zeros y1 at p.mNx which is the face at i=Nx-1 ONLY.
    # Wait, let me re-read: p.mNx is all (j,k) at i=Nx-1.
    # But in eval_f.m:  y1(p.mNx) = 0 — only the i=Nx-1 face!
    # Where does i=0 get zeroed?
    # y1 starts as zeros, then interior values are scattered at M2 (which
    # doesn't include i=0 or i=Nx). Then y1(p.mNx) = 0 zeros at i=Nx-1.
    # So the faces at i=0 stay zero (never set), and i=Nx stays zero (never set).
    # BUT i=Nx-1 IS set by the M2 scatter (it's an interior node), so it
    # needs to be explicitly zeroed.
    #
    # Wait, i=Nx-1 in 0-based (i_1=Nx in 1-based) — is that interior?
    # Interior = i_1 from 2 to Nx, i.e. 0-based i from 1 to Nx-1.
    # So i=Nx-1 IS the last interior layer in x.
    # MATLAB zeros y1 at that layer. Hmm...
    #
    # Actually in MATLAB:
    # mNx(h) = Ny + (Nx+1)*(j-1) + (Nx+1)*(Ny+1)*(k-1)
    # Since Nx=Ny in that specific run, Ny = Nx, so:
    # = Nx + mj*(j-1) + mk*(k-1)
    # In 0-based: (Nx) - 1 = Nx-1. So it's at i = Nx-1 (0-based).
    # But Nx-1 IS an interior node (interior is i=1..Nx-1).
    # The MATLAB code zeros the LAST interior x-layer of y1.
    #
    # Actually wait — let me re-read more carefully.
    # MATLAB: mNx(h_x) = Ny + (p.Nx+1)*(j-1)+(p.Nx+1)*(p.Ny+1)*(k-1)
    # Here Ny is the NUMBER, not Ny+1. In MATLAB 1-based indexing:
    # i_1 = Ny means the i-index = Ny (1-based).
    # When Nx=Ny, i_1 = Ny = Nx, which is i_0 = Nx-1 (0-based).
    # But for general Nx≠Ny: i_1 = Ny... that's wrong for a general case.
    # 
    # Wait, that's a BUG in the MATLAB code — or rather, it's only used
    # when Nx = Ny (square grid). Let me look again...
    # Actually the MATLAB line is: p.mNx(h_x) = Ny + ...
    # But Ny here is Matlab variable p.Ny, which is the Ny parameter.
    # The x index is supposed to be i_1 = Nx (1-based) = Nx-1 (0-based).
    # But the code writes Ny, not Nx! 
    # This means p.mNx actually stores indices at i_1 = Ny (1-based).
    #
    # If Nx != Ny, this is wrong. For Nx=Ny it works. 
    # This is a bug in the MATLAB code that only works for square grids.

    print(f"\n  Normal BC masks:")
    print(f"    x_normal_bc_mask: {len(idx.x_normal_bc_mask)} entries (faces at i=0 and i={Nx-1})")
    print(f"    y_normal_bc_mask: {len(idx.y_normal_bc_mask)} entries (faces at j=0 and j={Ny-1})")
    print(f"    z_normal_bc_mask: {len(idx.z_normal_bc_mask)} entries (faces at k=0 and k={Nz-1})")

    # Verify: x_normal_bc_mask should contain exactly the union of
    # face at i=0 and face at i=Nx-1 (for ALL j, k)
    expected_x_mask = set()
    for k_0 in range(Nz + 1 if Nz > 1 else 1):
        for j_0 in range(Ny + 1):
            expected_x_mask.add(0 + (Nx + 1) * j_0 + (Nx + 1) * (Ny + 1) * k_0)  # i=0
            expected_x_mask.add((Nx - 1) + (Nx + 1) * j_0 + (Nx + 1) * (Ny + 1) * k_0)  # i=Nx-1
    actual_x_mask = set(idx.x_normal_bc_mask)
    if actual_x_mask == expected_x_mask:
        print(f"    x_normal_bc_mask: ✓ matches expected (i=0, i={Nx-1})")
    else:
        print(f"    x_normal_bc_mask: ✗ MISMATCH")
        all_ok = False

    if all_ok:
        print(f"\n  ✓ ALL INDICES MATCH for Nx={Nx}, Ny={Ny}, Nz={Nz}")
    else:
        print(f"\n  ✗ SOME INDICES MISMATCH for Nx={Nx}, Ny={Ny}, Nz={Nz}")

    return all_ok


def verify_applied_field():
    """Verify the applied field is uniform in z."""
    print(f"\n{'='*60}")
    print("Verifying applied magnetic field")
    print(f"{'='*60}")

    Nx, Ny, Nz = 10, 10, 4
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz)
    idx = construct_indices(params)

    Bz_applied = 3.0
    Bx_vec, By_vec, Bz_vec = build_boundary_field_vectors(0.0, 0.0, Bz_applied, params, idx)

    print(f"\n  Applied Bz = {Bz_applied}")
    print(f"  Bz_vec: nonzero entries = {np.count_nonzero(Bz_vec)}")
    print(f"  Bz_vec unique nonzero values = {np.unique(Bz_vec[Bz_vec != 0])}")

    # Where are the nonzero entries?
    nz_indices = np.where(Bz_vec != 0)[0]
    print(f"  Nonzero Bz indices ({len(nz_indices)} total):")

    # Check: for Bz applied uniformly, the boundary condition indices should be
    # classicBz = [m1x_int, m1y_int, mNxp1_int, mNyp1_int]
    # which is: x_face_lo_inner, y_face_lo_inner, x_face_hi_inner, y_face_hi_inner
    expected_bz_idx = np.concatenate([
        idx.x_face_lo_inner, idx.y_face_lo_inner,
        idx.x_face_hi_inner, idx.y_face_hi_inner,
    ])
    expected_set = set(expected_bz_idx)
    actual_set = set(nz_indices)

    # Due to np.add.at, some indices may overlap (appear in multiple face lists)
    # Let's check the values
    print(f"\n  Expected Bz placed at {len(expected_set)} unique boundary nodes")
    print(f"  Actual nonzero at {len(actual_set)} nodes")

    if actual_set == expected_set:
        print("  ✓ Bz boundary indices match expected")
    else:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if missing:
            print(f"  ✗ Missing {len(missing)} indices: {sorted(missing)[:10]}...")
        if extra:
            print(f"  ✗ Extra {len(extra)} indices: {sorted(extra)[:10]}...")

    # Check for uniformity of Bz values at the boundary
    vals_at_bdy = Bz_vec[sorted(actual_set)]
    print(f"\n  Bz values at boundary: min={vals_at_bdy.min():.4f}, "
          f"max={vals_at_bdy.max():.4f}, unique={np.unique(vals_at_bdy)}")
    if len(np.unique(vals_at_bdy)) == 1:
        print("  ✓ Bz is uniform at all boundary nodes")
    else:
        print("  ✗ Bz is NOT uniform — some nodes have double-counted values")
        # Show which indices have non-standard values
        for v in np.unique(vals_at_bdy):
            count = np.sum(vals_at_bdy == v)
            print(f"      Bz={v:.4f}: {count} nodes")

    # Verify Bx and By are zero
    print(f"\n  Bx_vec all zero: {np.allclose(Bx_vec, 0)}")
    print(f"  By_vec all zero: {np.allclose(By_vec, 0)}")


def verify_boundary_conditions():
    """Verify boundary condition symmetry."""
    print(f"\n{'='*60}")
    print("Verifying boundary condition symmetry")
    print(f"{'='*60}")

    Nx, Ny, Nz = 10, 10, 4
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz)
    idx = construct_indices(params)

    # Check that face indices are symmetric (same count on lo and hi)
    print(f"\n  x face lo inner: {len(idx.x_face_lo_inner)} nodes")
    print(f"  x face hi inner: {len(idx.x_face_hi_inner)} nodes")
    print(f"  x first inner:   {len(idx.x_first_inner)} nodes")
    print(f"  x last inner:    {len(idx.x_last_inner)} nodes")
    print(f"  y face lo inner: {len(idx.y_face_lo_inner)} nodes")
    print(f"  y face hi inner: {len(idx.y_face_hi_inner)} nodes")
    print(f"  y first inner:   {len(idx.y_first_inner)} nodes")
    print(f"  y last inner:    {len(idx.y_last_inner)} nodes")
    print(f"  z face lo inner: {len(idx.z_face_lo_inner)} nodes")
    print(f"  z face hi inner: {len(idx.z_face_hi_inner)} nodes")
    print(f"  z first inner:   {len(idx.z_first_inner)} nodes")
    print(f"  z last inner:    {len(idx.z_last_inner)} nodes")

    # Check the MATLAB bug: in contruct_indices.m, the x_face condition
    # uses "j != 1 && j != p.Nx+1" instead of "j != 1 && j != p.Ny+1"
    # This means the inner x face indices are restricted to j in [2, Nx]
    # instead of j in [2, Ny]. If Nx != Ny, this is wrong.
    print(f"\n  MATLAB BUG CHECK:")
    print(f"  MATLAB x inner: j != 1 && j != Nx+1 → j in [2,{Nx}] (should be [2,{Ny}])")
    print(f"  MATLAB y inner: i != 1 && i != Ny+1 → i in [2,{Ny}] (should be [2,{Nx}])")
    if Nx == Ny:
        print(f"  With Nx=Ny={Nx}, the bug is invisible")
    else:
        print(f"  ✗ With Nx≠Ny, the MATLAB code has wrong inner face counts!")

    # Also check: MATLAB mNx = Ny + ... uses Ny for the x-index!
    # This gives i_1 = Ny (MATLAB 1-based) = i_0 = Ny-1 (Python 0-based)
    # The correct value should be i_0 = Nx-1.
    # Our Python code uses i_last = Nx-1, which is correct.
    print(f"\n  MATLAB mNx index: i_1 = Ny = {Ny} (should be Nx = {Nx})")
    if Nx == Ny:
        print(f"  With Nx=Ny, mNx gives i_0={Ny-1} = correct Nx-1={Nx-1}")
    else:
        print(f"  ✗ mNx gives i_0={Ny-1} but should be i_0={Nx-1}")

    # Verify that our Python normal BC masks are at the right indices
    print(f"\n  Checking normal BC mask contents:")
    mj = Nx + 1
    mk = (Nx + 1) * (Ny + 1)

    # x_normal_bc_mask should be at i=0 and i=Nx-1
    x_mask_i_vals = idx.x_normal_bc_mask % mj  # extract i coordinate
    x_unique_i = np.unique(x_mask_i_vals)
    print(f"  x_normal_bc_mask: i values = {x_unique_i} (expect [0, {Nx-1}])")
    if set(x_unique_i) == {0, Nx - 1}:
        print(f"  ✓ x_normal_bc_mask at correct i planes")
    else:
        print(f"  ✗ x_normal_bc_mask at wrong i planes!")

    # y_normal_bc_mask should be at j=0 and j=Ny-1
    y_mask_j_vals = (idx.y_normal_bc_mask % mk) // mj  # extract j coordinate
    y_unique_j = np.unique(y_mask_j_vals)
    print(f"  y_normal_bc_mask: j values = {y_unique_j} (expect [0, {Ny-1}])")
    if set(y_unique_j) == {0, Ny - 1}:
        print(f"  ✓ y_normal_bc_mask at correct j planes")
    else:
        print(f"  ✗ y_normal_bc_mask at wrong j planes!")


def verify_symmetry_test():
    """Run a small simulation with Bz field and check for C4 symmetry."""
    print(f"\n{'='*60}")
    print("Symmetry test: uniform Bz on square film should give C4-symmetric |ψ|²")
    print(f"{'='*60}")

    import tdgl3d

    Nx = Ny = 10
    Nz = 2
    params = SimulationParameters(Nx=Nx, Ny=Ny, Nz=Nz, kappa=2.0)
    field = AppliedField(Bz=2.0, t_on_fraction=0.7)
    device = tdgl3d.Device(params, applied_field=field)

    # Uniform initial condition (no random perturbation)
    x0 = tdgl3d.StateVector.uniform_superconducting(params).data.copy()

    dt = 0.01
    t_stop = 10.0
    solution = tdgl3d.solve(
        device, t_start=0.0, t_stop=t_stop, dt=dt,
        method="euler", x0=x0, save_every=50, progress=False,
    )

    psi = solution.psi(step=-1).reshape(Nx - 1, Ny - 1, max(Nz - 1, 1))
    psi2 = np.abs(psi) ** 2
    mid_z = max(Nz - 1, 1) // 2
    slice_z = psi2[:, :, mid_z]

    print(f"\n  |ψ|² at mid-z plane:")
    print(f"    mean = {np.mean(slice_z):.6f}")
    print(f"    min  = {np.min(slice_z):.6f}")
    print(f"    max  = {np.max(slice_z):.6f}")

    # Check C4 symmetry: should be unchanged under 90° rotation
    rot90 = np.rot90(slice_z)
    diff = np.max(np.abs(slice_z - rot90))
    print(f"\n  C4 symmetry check (uniform IC, no perturbation):")
    print(f"    max|ψ² - rot90(ψ²)| = {diff:.2e}")
    if diff < 1e-10:
        print(f"    ✓ Perfect C4 symmetry (as expected with uniform IC)")
    else:
        print(f"    ✗ C4 symmetry broken! Something is wrong in BCs or indices")

    # Check x↔y mirror symmetry
    mirror = slice_z.T
    diff_mirror = np.max(np.abs(slice_z - mirror))
    print(f"    max|ψ² - transpose(ψ²)| = {diff_mirror:.2e}")
    if diff_mirror < 1e-10:
        print(f"    ✓ Perfect mirror symmetry")
    else:
        print(f"    ✗ Mirror symmetry broken!")


if __name__ == "__main__":
    # Test several grid sizes
    ok_all = True
    for Nx, Ny, Nz in [(4, 4, 2), (6, 6, 3), (10, 10, 4), (8, 12, 3)]:
        ok = verify_indices(Nx, Ny, Nz)
        if not ok:
            ok_all = False

    verify_applied_field()
    verify_boundary_conditions()
    verify_symmetry_test()

    print(f"\n{'='*60}")
    if ok_all:
        print("ALL INDEX CHECKS PASSED")
    else:
        print("SOME INDEX CHECKS FAILED")
    print(f"{'='*60}")
