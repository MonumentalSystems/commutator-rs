import tempfile
from pathlib import Path
import unittest

from inspect_wannier import inspect, read_section
from split_tb import split
from io import StringIO
from unittest.mock import patch


WIN = "num_wann=2\nspinors=.true.\nbegin unit_cell_cart\nAng\n1 0 0\n0 1 0\n0 0 1\nend unit_cell_cart\nbegin atoms_cart\nAng\nTe 0 0 0\nend atoms_cart\n"
TB = "fixture\n1 0 0\n0 1 0\n0 0 1\n2\n1\n1\n0 0 0\n1 1 2.0 0\n2 1 1 0.5\n1 2 1 -0.5\n2 2 3 0\n0 0 0\n1 1 0.2 0 0.3 0 0.4 0\n2 1 0 0.7 0 0 0 0\n1 2 0 -0.7 0 0 0 0\n2 2 0.8 0 0.9 0 1.0 0\n"


class WannierInspectionTests(unittest.TestCase):
    def test_rejects_duplicate_orbital_pair(self):
        with self.assertRaisesRegex(ValueError, "Duplicate orbital"):
            read_section(StringIO("0 0 0\n1 1 0 0\n1 1 1 0\n2 1 1 0\n2 2 1 0\n"), 2, 1, 1, True)

    def test_rejects_truncated_data(self):
        with self.assertRaisesRegex(ValueError, "end of file"):
            read_section(StringIO("0 0 0\n1 1 0 0\n"), 2, 1, 1, True)

    def test_rejects_nonfinite_value(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            read_section(StringIO("0 0 0\n1 1 nan 0\n"), 1, 1, 1, True)

    def test_split_preserves_offdiagonal_position_and_basis(self):
        # Imaginary off-diagonal r elements must survive, not be replaced by centres.
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / "test_tb.dat"
            source.write_text(TB)
            win = root / "input.win"
            win.write_text(WIN)
            result = split(source, root / "out", "fixture", win)
            self.assertEqual(result["num_wann"], 2)
            self.assertEqual(result["position_hermiticity_max_abs_angstrom"], 0)
            hr = inspect(root / "out/fixture_hr.dat", "hr")
            self.assertEqual(hr["hamiltonian_hermiticity_max_abs_ev"], 0)
            self.assertIn("0 0 0 2 1 0 0.7", (root / "out/fixture_r.dat").read_text())
            self.assertIn("X 0.20000000000000001", (root / "out/fixture_centres.xyz").read_text())
            with self.assertRaises(FileExistsError):
                split(source, root / "out", "fixture", win)

    def test_invalid_win_leaves_no_artifacts(self):
        for invalid in (WIN.replace("num_wann=2", "num_wann=3"),
                        WIN.replace("1 0 0", "2 0 0"),
                        WIN.replace("Ang", "Bohr"),
                        WIN.replace("Te 0 0 0", "Te 0 0")):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
                root = Path(temporary)
                source, win = root / "input_tb.dat", root / "input.win"
                source.write_text(TB)
                win.write_text(invalid)
                with self.assertRaises(ValueError):
                    split(source, root / "out", "fixture", win)
                self.assertEqual({p.name for p in root.iterdir()}, {"input_tb.dat", "input.win"})

    def test_write_failure_cleans_owned_staging_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source, win = root / "input_tb.dat", root / "input.win"
            source.write_text(TB)
            win.write_text(WIN)
            with patch("split_tb.shutil.copyfile", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    split(source, root / "out", "fixture", win)
            self.assertEqual({p.name for p in root.iterdir()}, {"input_tb.dat", "input.win"})


if __name__ == "__main__":
    unittest.main()
