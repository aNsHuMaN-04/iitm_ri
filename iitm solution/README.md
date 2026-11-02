# mtj_auto - natural-language heterostructure builder for SIESTA

    python mtj_auto.py "Fe 8 layers / MgO 4 layers / Fe 8 layers, gap 2.2" --label FeMgOFe
    python mtj_auto.py "Fe(6) MgO(5) Fe(6), 2x2 supercell, vacuum 8" --poscar
    python mtj_auto.py --list
    python test_mtj_auto.py

Needs only NumPy. Output: `<label>.fdf` (SIESTA) and optionally `<label>.POSCAR`.

Prompt keywords: material names (see --list), layer counts ("8 layers of Fe", "Fe 8 layers", "Fe(8)"),
`gap <A>`, `NxM supercell`, `vacuum <A>`.

Pipeline: parse_prompt -> validate -> build_stack -> check_geometry -> write_fdf.
The .fdf has lattice, species, coordinates and DM.InitSpin; pseudopotentials, basis and k-points
must still be added before running SIESTA.
