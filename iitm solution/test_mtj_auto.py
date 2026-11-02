"""Run:  python test_mtj_auto.py   (plain asserts, no pytest needed)"""
import os, tempfile
import numpy as np
import mtj_auto as m


def test_parser_variants():
    for txt in ["Fe 8 layers / MgO 4 layers / Fe 8 layers",
                "8 layers of Fe, 4 layers of MgO, 8 layers of Fe",
                "Fe(8) MgO(4) Fe(8)"]:
        assert m.parse_prompt(txt).stack == [("fe", 8), ("mgo", 4), ("fe", 8)], txt


def test_options():
    s = m.parse_prompt("Fe 4 layers / MgO 2 layers / Fe 4 layers, gap 3.0, 2x3 supercell, vacuum 12")
    assert s.gaps == [3.0, 3.0] and (s.nx, s.ny) == (2, 3) and s.vacuum == 12.0


def test_layer_counts_match_notebooks():
    # notebooks: MgO 4 layers -> 16 atoms, Fe 16 layers -> 32 atoms, Mn3Ge 20 layers -> 40 atoms
    for key, n, atoms in [("mgo", 4, 16), ("fe", 16, 32), ("mn3ge", 20, 40)]:
        pos, sp, mg = m.build_block(key, n)
        assert len(sp) == atoms, (key, len(sp))


def test_odd_layer_trimming():
    pos, sp, _ = m.build_block("mgo", 3)            # 3 planes, not a whole number of cells
    assert len(m._planes(pos[:, 2])) == 3


def test_fe_layer_spacing():
    pos, _, _ = m.build_block("fe", 4)
    z = m._planes(pos[:, 2])
    assert abs((z[1] - z[0]) - 1.4301235) < 1e-6    # matches Fe_L notebook output


def test_supercell_atom_count():
    base = m.run("Fe 4 layers / MgO 2 layers / Fe 4 layers", quiet=True)[2]
    big = m.run("Fe 4 layers / MgO 2 layers / Fe 4 layers, 2x2 supercell", quiet=True)[2]
    assert len(big) == 4 * len(base)


def test_unknown_material_error():
    try:
        m.parse_prompt("build some unobtainium")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp())
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
