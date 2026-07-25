#!/bin/bash
# Quick Start Guide for tdgl3d Physics Validation
# Run this script to see recommended workflow commands

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  tdgl3d — Quick Start Guide (Physics Validation Workflow)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if HDF5 file exists
HDF5_FILE="sis_square_Bz0.50_t120.h5"

if [ -f "$HDF5_FILE" ]; then
    echo "✓ Found saved simulation: $HDF5_FILE"
    echo ""
    echo "Available commands:"
    echo ""
    echo "  1. Visualize saved results (instant):"
    echo "     python3 examples/visualize_solution.py $HDF5_FILE"
    echo ""
    echo "  2. Custom plots:"
    echo "     python3 examples/visualize_solution.py $HDF5_FILE \\"
    echo "         --plots psi phase current bfield --step 100 --dpi 300"
    echo ""
    echo "  3. Different z-slice (top layer):"
    echo "     python3 examples/visualize_solution.py $HDF5_FILE --slice-z 20"
    echo ""
    echo "  4. Load and analyze in Python:"
    echo "     from tdgl3d import Solution"
    echo "     solution = Solution.load('$HDF5_FILE')"
    echo ""
else
    echo "⚠ No saved simulation found."
    echo ""
    echo "To run your first simulation (~30 minutes):"
    echo ""
    echo "  python3 examples/sis_square_with_hole.py"
    echo ""
    echo "This will:"
    echo "  • Run S/I/S trilayer simulation with rectangular hole"
    echo "  • Auto-save to: $HDF5_FILE"
    echo "  • Generate initial plots"
    echo "  • Print vortex counts and physics diagnostics"
    echo ""
    echo "After that, you can visualize instantly anytime!"
    echo ""
fi

echo "─────────────────────────────────────────────────────────────────"
echo "Quick Tests (< 1 second each):"
echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "  • Fast smoke tests (0.2s):"
echo "    python3 -m pytest tests/test_hole_bc_smoke.py -v"
echo ""
echo "  • Physics validation (3s):"
echo "    python3 -m pytest tests/test_physics_nonz_fields.py -v"
echo ""
echo "  • All tests (14s):"
echo "    python3 -m pytest tests/ -q"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "For detailed workflow guide, see:"
echo "  examples/WORKFLOW.md"
echo "═══════════════════════════════════════════════════════════════"
echo ""
