import ringo
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

import os
import shutil
import glob
import logging
import subprocess
import tempfile

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from .config import LevelOfTheoryConfig
from .utils import (
    get_xyzs_from_rdmol, )


class MoleculeOptimizer(ABC):

    def __init__(self,
                 config: LevelOfTheoryConfig,
                 distance_constraints: list[tuple[int, int, float]] = []):
        super().__init__()
        self.config = config
        self.distance_constraints = distance_constraints

    def _coords_are_valid(self, xyz: np.ndarray) -> bool:
        if len(self.distance_constraints) == 0:
            return True
        return all(
            abs(np.linalg.norm(xyz[a] - xyz[b]) - expected_value) < 0.1
            for a, b, expected_value in self.distance_constraints)

    @abstractmethod
    def optimize(
        self,
        mol: Chem.Mol,
        xyzs: np.ndarray,
        dihedral_constraints: List[Tuple[Tuple[int, int, int, int],
                                         float]] = [],
    ) -> Tuple[np.ndarray, float]:
        ...

    @abstractmethod
    def single_point_energy(self, mol: Chem.Mol) -> float:
        ...


class RDKitMoleculeOptimizer(MoleculeOptimizer):

    def optimize(
        self,
        mol: Chem.Mol,
        xyzs: np.ndarray,
        dihedral_constraints: List[Tuple[Tuple[int, int, int, int],
                                         float]] = []
    ) -> Tuple[np.ndarray, float]:
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, xyzs[atom_idx])
        mol.AddConformer(conf)

        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        if dihedral_constraints:
            for atoms, value in dihedral_constraints:
                ff.MMFFAddTorsionConstraint(
                    *atoms,
                    False,
                    float(-value * ringo.RAD2DEG),
                    float(-value * ringo.RAD2DEG),
                    1e4,
                )
        for a, b, value in self.distance_constraints:
            ff.MMFFAddDistanceConstraint(
                a,
                b,
                False,
                value,
                value,
                1e4,
            )

        try_i = 0
        return_code = 1
        while try_i < 10 and return_code != 0:
            return_code = ff.Minimize(maxIts=10000)
            try_i += 1
        assert return_code == 0

        energy = ff.CalcEnergy()
        opt_xyzs = get_xyzs_from_rdmol(mol)
        assert self._coords_are_valid(opt_xyzs)
        return opt_xyzs, energy

    def single_point_energy(self, mol: Chem.Mol) -> float:
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        energy = ff.CalcEnergy()
        return energy


class XTBMoleculeOptimizer(MoleculeOptimizer, ABC):

    def _write_xyz(self, path: Path, mol: Chem.Mol, xyzs: np.ndarray):
        with open(path, "w") as f:
            f.write(f"{mol.GetNumAtoms()}\n\n")
            for i, atom in enumerate(mol.GetAtoms()):
                sym = atom.GetSymbol()
                x, y, z = xyzs[i]
                f.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")

    def _write_control_file(
        self,
        path: Path,
        constraints: List[Tuple[Tuple[int, int, int, int], float]],
        add_mtd=False,
    ):
        if len(constraints) == 0 and len(self.distance_constraints) == 0:
            return

        lines = ["$constrain"]
        for atoms, value in constraints:
            a1, a2, a3, a4 = [i + 1 for i in atoms]
            lines.append(
                f"    dihedral: {a1},{a2},{a3},{a4},{-np.degrees(value):.6f}")
        for a, b, value in self.distance_constraints:
            lines.append(f"    distance: {a+1},{b+1},{value:.4f}")
        lines.append("    force constant=5.0")
        lines.append("$end")
        lines.append(f"""
$md
  hmass=5
  time=     20.00
  temp=    300.00
  step=      1.50
  shake=2
  dump=100
  skip=1
$set
  mddump  1000
$end
$metadyn
  save=57
  kpush=0.01
  alp=0.280800
$end
""")

        with open(path, "w") as f:
            f.write('\n'.join(lines))

    def _read_optimized_xyz(self, path: Path, natoms: int) -> np.ndarray:
        with open(path, "r") as f:
            lines = f.readlines()[2:2 + natoms]
        xyz = np.array([[float(x) for x in line.strip().split()[1:4]]
                        for line in lines])
        return xyz

    def optimize(
        self,
        mol: Chem.Mol,
        xyzs: np.ndarray,
        dihedral_constraints: List[Tuple[Tuple[int, int, int, int],
                                         float]] = []
    ) -> Tuple[np.ndarray, float]:
        xyzs, preopt_energy = RDKitMoleculeOptimizer(
            None, self.distance_constraints).optimize(
                mol, xyzs,
                dihedral_constraints)
        with tempfile.TemporaryDirectory() as tmpdir:
            # tmpdir = os.path.join(os.getcwd(), 'xtbopt')
            # if os.path.isdir(tmpdir):
            #     shutil.rmtree(tmpdir)
            # os.mkdir(tmpdir)

            logging.info(f"Using tempdir for XTB opt calculation: {tmpdir}")
            tmpdir = Path(tmpdir)
            xyz_path = tmpdir / "inputgeom.xyz"
            control_path = tmpdir / "inputgeom.control"

            self._write_xyz(xyz_path, mol, xyzs)
            self._write_control_file(control_path, dihedral_constraints)

            cmd = [
                self.config.executable,
                str(xyz_path),
                "--opt",
                "-I",
                str(control_path),
                *self.config.flags.split(),
            ]

            result = subprocess.run(cmd,
                                    cwd=tmpdir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)

            with open(os.path.join(tmpdir, "output.log"), "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    f"xTB optimization failed:\n{result.stderr}")

            opt_path = tmpdir / "xtbopt.xyz"
            energy = self._parse_final_energy(result.stdout)
            opt_xyz = self._read_optimized_xyz(opt_path, mol.GetNumAtoms())
            assert self._coords_are_valid(opt_xyz)

        return opt_xyz, energy

    def single_point_energy(self, mol: Chem.Mol) -> float:
        with tempfile.TemporaryDirectory() as tmpdir:
            # tmpdir = './xtbopt'
            # if os.path.isdir(tmpdir):
            #     shutil.rmtree(tmpdir)
            # os.mkdir(tmpdir)
            logging.info(
                f"Using tempdir for XTB single-point calculation: {tmpdir}")
            tmpdir = Path(tmpdir)
            xyz_path = tmpdir / "inputgeom.xyz"

            conf = mol.GetConformer()
            xyzs = np.array([
                list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
            ])
            self._write_xyz(xyz_path, mol, xyzs)

            cmd = [
                self.config.executable,
                str(xyz_path),
                *self.config.flags.split(),
            ]

            result = subprocess.run(cmd,
                                    cwd=tmpdir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)

            with open(os.path.join(tmpdir, "output.log"), "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    f"xTB single-point failed:\n{result.stderr}")

            resulting_energy = self._parse_final_energy(result.stdout)
        return resulting_energy

    def _parse_final_energy(self, stdout: str) -> float:
        for line in stdout.splitlines():
            if "TOTAL ENERGY" in line:
                return float(line.split()[3]) * ringo.H2KC
        raise ValueError("Could not parse energy from xTB output.")

    def run_mtd(self, mol: Chem.Mol, xyzs: np.ndarray):
        with tempfile.TemporaryDirectory() as tmpdir:
            # tmpdir = os.path.join(os.getcwd(), 'xtbopt')
            # if os.path.isdir(tmpdir):
            #     shutil.rmtree(tmpdir)
            # os.mkdir(tmpdir)

            logging.info(f"Using tempdir for XTB MTD calculation: {tmpdir}")
            tmpdir = Path(tmpdir)
            xyz_path = tmpdir / "inputgeom.xyz"
            control_path = tmpdir / "inputgeom.control"

            self._write_xyz(xyz_path, mol, xyzs)
            self._write_control_file(control_path, [])

            cmd = [
                self.config.executable,
                str(xyz_path),
                "--md",
                "-I",
                str(control_path),
                *self.config.flags.split(),
            ]

            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with open(os.path.join(tmpdir, "output.log"), "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    f"xTB optimization failed:\n{result.stderr}")

            xyzs = [
                read_coord_file(snap_file)
                for snap_file in glob.glob(str(tmpdir / "scoord.*"))
            ]
        return xyzs


BOHR2A = 0.529177


def fix_atom_symbol(symbol):
    return symbol[0].upper() + symbol[1:].lower()


def read_coord_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    coords = []
    for line in lines[1:]:
        if line.startswith('$'):
            break
        tokens = line.split()
        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
        coord = np.array([x, y, z])
        coords.append(coord)

    n_atoms = len(coords)
    xyz_matr = np.zeros((n_atoms, 3))
    for i, xyz in enumerate(coords):
        xyz_matr[i, :] = xyz * BOHR2A
    return xyz_matr


LEVELS_OF_THEORY: dict[str, MoleculeOptimizer] = {
    'rdkit': RDKitMoleculeOptimizer,
    'xtb': XTBMoleculeOptimizer,
}
