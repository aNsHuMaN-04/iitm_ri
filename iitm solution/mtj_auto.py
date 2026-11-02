#!/usr/bin/env python3
"""
mtj_auto.py - Natural-language driven heterostructure builder for SIESTA.

    python mtj_auto.py "Fe 8 layers / MgO 4 layers / Fe 8 layers, gap 2.2, 2x2 supercell"

Pipeline (same stages as the notebooks, now in one module):
    1. parse_prompt()      plain text  -> BuildSpec (materials, layers, gaps, supercell)
    2. validate()          lattice-mismatch and sanity checks
    3. build_stack()       layer generators -> stacked atoms (numpy, no heavy deps)
    4. check_geometry()    minimum interatomic distance check
    5. write_fdf()         SIESTA input file (+ optional POSCAR)

Conventions (from the notebooks):
    * a "layer" is one distinct atomic plane along z
    * all blocks share one in-plane cell a = b = A_COMMON (Angstrom)
    * "dimension": 1D = stack along z only, 2D = repeat nx x ny in-plane
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

A_COMMON = 4.045            # shared in-plane lattice constant (Angstrom)
PLANE_TOL = 1e-3            # tolerance when grouping atoms into planes (Angstrom)
MIN_DIST_WARN = 1.5         # flag atoms closer than this (Angstrom)

ATOMIC_NUMBER = {"Ge": 32, "Fe": 26, "Mg": 12, "O": 8, "Mn": 25, "Cr": 24,
                 "Ni": 28, "Co": 27, "Ga": 31}
SPECIES_ORDER = ["Ge", "Fe", "Mg", "O", "Mn", "Cr", "Ni", "Co", "Ga"]  # fixed -> consistent labels


# ----------------------------------------------------------------------------
# 1. Material library: one generator per crystal prototype
# ----------------------------------------------------------------------------
@dataclass
class Block:
    """Atoms of one unit cell: fractional coords, species, moments."""
    frac: np.ndarray            # (n,3) fractional coordinates
    species: List[str]
    magmom: List[float]
    c: float                    # cell height (Angstrom)
    layers_per_cell: int


@dataclass
class Material:
    name: str
    a_natural: float            # in-plane constant this material "wants" (Angstrom)
    make_cell: Callable[[], Block]
    default_layers: int = 4


def _bcc_rotated(el: str, a_bcc: float, mu: float) -> Material:
    """bcc metal in the 45-degree rotated 4-atom cell (a = sqrt(2) * a_bcc), as in Fe_L."""
    def make() -> Block:
        frac = np.array([[0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
        frac[:, :2] = (frac[:, :2] + 0.25) % 1.0                  # dx = dy = 0.25 as in notebook
        return Block(frac, [el] * 4, [mu] * 4, a_bcc, 2)
    return Material(el, a_bcc * math.sqrt(2), make, default_layers=8)


def _rocksalt(cat: str, an: str, a_bulk: float) -> Material:
    """Rock-salt AB: cation 4a, anion 4b, cubic conventional cell (as in MgO_1)."""
    def make() -> Block:
        frac = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                         [.5, .5, .5], [0, 0, .5], [0, .5, 0], [.5, 0, 0]])
        return Block(frac, [cat] * 4 + [an] * 4, [0.0] * 8, A_COMMON, 2)
    return Material(cat + an, a_bulk, make, default_layers=4)


def _heusler(x: str, z: str, a: float, c: float, mu2b: float, mu4d: float) -> Material:
    """Inverse Heusler X3Z (tetragonal), 8 atoms / 4 layers, as in Mn3Ge."""
    def make() -> Block:
        frac = np.array([[0, 0, 0], [.5, .5, .5],                 # 2a  (Z)
                         [.5, .5, 0], [0, 0, .5],                 # 2b  (X)
                         [.5, 0, .25], [.5, 0, .75], [0, .5, .25], [0, .5, .75]])  # 4d (X)
        frac[:, :2] = (frac[:, :2] + 0.25) % 1.0
        return Block(frac, [z] * 2 + [x] * 6, [0.0] * 2 + [mu2b] * 2 + [mu4d] * 4, c, 4)
    return Material(x + "3" + z, a, make, default_layers=8)


LIBRARY: Dict[str, Material] = {m.name.lower(): m for m in [
    _bcc_rotated("Fe", 2.860247, 2.2),
    _bcc_rotated("Cr", 2.88, 0.0),
    _rocksalt("Mg", "O", 4.212),
    _rocksalt("Ni", "O", 4.177),
    _rocksalt("Co", "O", 4.26),
    _heusler("Mn", "Ge", 3.73, 5.9162, -3.4, 1.9),
    _heusler("Mn", "Ga", 3.90, 5.9162, -3.0, 2.0),
]}


# ----------------------------------------------------------------------------
# 2. Natural-language parser (rule-based, no external API)
# ----------------------------------------------------------------------------
@dataclass
class BuildSpec:
    stack: List[Tuple[str, int]]                 # [(material key, n_layers), ...]
    gaps: List[float]                            # len(stack) - 1
    nx: int = 1
    ny: int = 1
    vacuum: float = 10.0
    label: str = "MTJ"
    notes: List[str] = field(default_factory=list)


def parse_prompt(text: str, default_gap: float = 2.2) -> BuildSpec:
    """Turn 'Fe 8 layers / MgO 4 layers / Fe 8 layers, gap 2.2, 2x2 supercell' into a BuildSpec."""
    t = text.strip()
    names = "|".join(sorted((re.escape(k) for k in LIBRARY), key=len, reverse=True))
    # "8 layers of Fe"  |  "Fe 8 layers"  |  "Fe(8)"  |  bare "Fe"
    pat = re.compile(
        rf"(?:(?P<n1>\d+)\s*(?:layers?|ml|monolayers?)?\s*(?:of\s+)?(?P<m1>{names})\b)"
        rf"|(?:(?P<m2>{names})\b\s*[\(:x]?\s*(?P<n2>\d+)?\s*(?:layers?|ml)?\)?)", re.I)
    stack, notes = [], []
    for m in pat.finditer(t):
        key = (m.group("m1") or m.group("m2")).lower()
        n = m.group("n1") or m.group("n2")
        if n is None:
            n = LIBRARY[key].default_layers
            notes.append(f"no layer count for {LIBRARY[key].name}; using default {n}")
        stack.append((key, int(n)))
    if not stack:
        raise ValueError(f"No known materials found. Available: {', '.join(m.name for m in LIBRARY.values())}")

    gap = default_gap
    g = re.search(r"gaps?\s*(?:of|=|:)?\s*([\d.]+)", t, re.I)
    if g:
        gap = float(g.group(1))
    sc = re.search(r"(\d+)\s*[x×]\s*(\d+)(?:\s*[x×]\s*\d+)?\s*(?:super\s*cell|cell)?", t, re.I)
    nx, ny = (int(sc.group(1)), int(sc.group(2))) if sc else (1, 1)
    if re.search(r"\b1\s*-?\s*d(imension(al)?)?\b", t, re.I):
        nx = ny = 1
    vac = re.search(r"vacuum\s*(?:of|=|:)?\s*([\d.]+)", t, re.I)
    return BuildSpec(stack, [gap] * (len(stack) - 1), nx, ny,
                     float(vac.group(1)) if vac else 10.0, notes=notes)


# ----------------------------------------------------------------------------
# 3. Validation
# ----------------------------------------------------------------------------
def validate(spec: BuildSpec) -> List[str]:
    msgs = list(spec.notes)
    for key, n in spec.stack:
        mat = LIBRARY[key]
        if n < 1:
            raise ValueError(f"{mat.name}: layer count must be >= 1")
        strain = (A_COMMON - mat.a_natural) / mat.a_natural * 100
        flag = "  <-- large strain" if abs(strain) > 5 else ""
        msgs.append(f"{mat.name}: natural a = {mat.a_natural:.3f} A -> forced to {A_COMMON} A "
                    f"({strain:+.1f}% strain){flag}")
    if not (1 <= spec.nx <= 10 and 1 <= spec.ny <= 10):
        raise ValueError("supercell repeats must be between 1 and 10")
    return msgs


# ----------------------------------------------------------------------------
# 4. Builder
# ----------------------------------------------------------------------------
def _planes(z: np.ndarray) -> List[float]:
    """Distinct z-planes (sorted)."""
    out: List[float] = []
    for v in np.sort(z):
        if not out or v - out[-1] > PLANE_TOL:
            out.append(float(v))
    return out


def build_block(key: str, n_layers: int) -> Tuple[np.ndarray, List[str], List[float]]:
    """n_layers atomic planes of one material, cartesian coords with min z = 0."""
    mat = LIBRARY[key]
    cell = mat.make_cell()
    n_cells = math.ceil(n_layers / cell.layers_per_cell)           # same rule as notebooks
    pos, sp, mg = [], [], []
    for i in range(n_cells):
        cart = cell.frac * np.array([A_COMMON, A_COMMON, cell.c])
        cart[:, 2] += cell.c * i                                    # cascade along z
        pos.append(cart); sp += cell.species; mg += cell.magmom
    pos = np.vstack(pos)
    keep_planes = _planes(pos[:, 2])[:n_layers]                      # trim odd layer counts
    mask = np.array([any(abs(z - p) < PLANE_TOL for p in keep_planes) for z in pos[:, 2]])
    pos = pos[mask]
    sp = [s for s, k in zip(sp, mask) if k]
    mg = [m for m, k in zip(mg, mask) if k]
    pos[:, 2] -= pos[:, 2].min()
    return pos, sp, mg


def build_stack(spec: BuildSpec):
    all_pos, all_sp, all_mg, z0, report = [], [], [], 0.0, []
    for i, (key, n) in enumerate(spec.stack):
        pos, sp, mg = build_block(key, n)
        pos[:, 2] += z0
        all_pos.append(pos); all_sp += sp; all_mg += mg
        report.append((LIBRARY[key].name, n, len(sp), z0, pos[:, 2].max()))
        z0 = pos[:, 2].max() + (spec.gaps[i] if i < len(spec.gaps) else 0.0)
    pos = np.vstack(all_pos)
    # in-plane supercell
    tiles = [pos + np.array([ix * A_COMMON, iy * A_COMMON, 0.0])
             for ix in range(spec.nx) for iy in range(spec.ny)]
    pos = np.vstack(tiles)
    sp = all_sp * (spec.nx * spec.ny)
    mg = all_mg * (spec.nx * spec.ny)
    # sort bottom -> top (stable), as AtomicStructure did
    order = np.argsort(pos[:, 2], kind="stable")
    pos, sp, mg = pos[order], [sp[i] for i in order], [mg[i] for i in order]
    lattice = np.diag([A_COMMON * spec.nx, A_COMMON * spec.ny, pos[:, 2].max() + spec.vacuum])
    return lattice, pos, sp, mg, report


# ----------------------------------------------------------------------------
# 5. Geometry check (periodic in x,y; z is a slab with vacuum)
# ----------------------------------------------------------------------------
def check_geometry(lattice: np.ndarray, pos: np.ndarray) -> float:
    Lx, Ly = lattice[0, 0], lattice[1, 1]
    d = pos[:, None, :] - pos[None, :, :]
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


# ----------------------------------------------------------------------------
# 6. Writers
# ----------------------------------------------------------------------------
def write_fdf(path: str, label: str, lattice, pos, sp, mg) -> None:
    present = [e for e in SPECIES_ORDER if e in sp]                   # fixed order, only used species
    idx = {e: i + 1 for i, e in enumerate(present)}
    with open(path, "w") as f:
        f.write(f"SystemName          {label}\nSystemLabel         {label}\n\n")
        f.write("LatticeConstant      1.00 Ang\n%block LatticeVectors\n")
        for v in lattice:
            f.write(f"    {v[0]:.8f}    {v[1]:.8f}    {v[2]:.8f}\n")
        f.write("%endblock LatticeVectors\n\n")
        f.write(f"NumberOfSpecies     {len(present)}\nNumberOfAtoms       {len(sp)}\n\n")
        f.write("%block ChemicalSpeciesLabel\n")
        for e in present:
            f.write(f"    {idx[e]}    {ATOMIC_NUMBER[e]}    {e}\n")
        f.write("%endblock ChemicalSpeciesLabel\n\n")
        f.write("AtomicCoordinatesFormat Ang\n%block AtomicCoordinatesAndAtomicSpecies\n")
        for p, s in zip(pos, sp):
            f.write(f"    {p[0]:.8f}    {p[1]:.8f}    {p[2]:.8f}    {idx[s]}\n")
        f.write("%endblock AtomicCoordinatesAndAtomicSpecies\n\n")
        f.write("Spin                polarized\n%block DM.InitSpin\n")
        for i, m in enumerate(mg, 1):
            if m != 0.0:
                f.write(f"    {i}    {m:.3f}\n")
        f.write("%endblock DM.InitSpin\n")
        f.write("\n# NOTE: pseudopotentials (.psf), basis and k-points must be added before running SIESTA.\n")


def write_poscar(path: str, label: str, lattice, pos, sp) -> None:
    order = [e for e in SPECIES_ORDER if e in sp]
    with open(path, "w") as f:
        f.write(f"{label}\n1.0\n")
        for v in lattice:
            f.write(f"  {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        f.write("  " + " ".join(order) + "\n  " + " ".join(str(sp.count(e)) for e in order) + "\nCartesian\n")
        for e in order:
            for p, s in zip(pos, sp):
                if s == e:
                    f.write(f"  {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def run(prompt: str, label: str = "MTJ", poscar: bool = False, quiet: bool = False):
    spec = parse_prompt(prompt)
    spec.label = label
    msgs = validate(spec)
    lattice, pos, sp, mg, report = build_stack(spec)
    dmin = check_geometry(lattice, pos)
    write_fdf(f"{label}.fdf", label, lattice, pos, sp, mg)
    if poscar:
        write_poscar(f"{label}.POSCAR", label, lattice, pos, sp)
    if not quiet:
        print("=" * 64)
        print(f"Prompt : {prompt}")
        print("Parsed :", " / ".join(f"{LIBRARY[k].name}({n})" for k, n in spec.stack),
              f"| gaps {spec.gaps} A | supercell {spec.nx}x{spec.ny}")
        for m in msgs:
            print("  check:", m)
        print("-" * 64)
        for name, n, cnt, z_lo, z_hi in report:
            print(f"  {name:<6} {n:>2} layers   z = {z_lo:7.3f} .. {z_hi:7.3f} A")
        print("-" * 64)
        print(f"Atoms: {len(sp)}   Cell: {lattice[0,0]:.3f} x {lattice[1,1]:.3f} x {lattice[2,2]:.3f} A")
        status = "OK" if dmin >= MIN_DIST_WARN else "WARNING: atoms too close"
        print(f"Min interatomic distance: {dmin:.3f} A  [{status}]")
        print(f"Wrote {label}.fdf" + (f" and {label}.POSCAR" if poscar else ""))
        print("=" * 64)
    return lattice, pos, sp, mg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Natural-language heterostructure builder for SIESTA")
    ap.add_argument("prompt", nargs="?", help="e.g. 'Fe 8 layers / MgO 4 layers / Fe 8 layers, gap 2.2'")
    ap.add_argument("--label", default="MTJ", help="SystemLabel / output file stem")
    ap.add_argument("--poscar", action="store_true", help="also write a POSCAR")
    ap.add_argument("--list", action="store_true", help="list known materials")
    a = ap.parse_args(argv)
    if a.list:
        for m in LIBRARY.values():
            print(f"{m.name:<6} natural a = {m.a_natural:.3f} A, default layers = {m.default_layers}")
        return 0
    if not a.prompt:
        ap.error("give a prompt, e.g. \"Fe 8 layers / MgO 4 layers / Fe 8 layers\"")
    try:
        run(a.prompt, a.label, a.poscar)
    except ValueError as e:
        print("Error:", e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
