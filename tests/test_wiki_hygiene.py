"""Repository hygiene checks use Git's file inventory, not local cache contents."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wiki_lint import repository_hygiene


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def write(self, name, text="example"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_ignored_scratch_is_local_but_force_added_scratch_is_blocked(self):
        self.write(".gitignore", "output/\n")
        self.write("output/campaign/run.sh")
        self.assertEqual(repository_hygiene(self.root), ([], []))
        self.git("add", "-f", "output/campaign/run.sh")
        errors, _ = repository_hygiene(self.root)
        self.assertTrue(any("tracked despite ignore" in e for e in errors))

    def test_new_unrelated_draft_is_found_without_scanning_research_prose(self):
        self.write("drafts/doordash-proposal.tex")
        self.write("raw/papers/relevant.md", "A technical comparison mentions DoorDash.")
        errors, warnings = repository_hygiene(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("drafts/doordash-proposal.tex", errors[0])
        self.assertEqual(warnings, [])

    def test_research_assets_allowed_and_removed_files_skipped(self):
        for name in ["proposal/research.pdf", "raw/papers/paper.pdf", "bundles/source.tar.gz"]:
            self.write(name)
        removed = self.write("doordash-draft.tex")
        self.git("add", ".")
        removed.unlink()
        self.assertEqual(repository_hygiene(self.root), ([], []))

    def test_placeholders_and_large_files_require_review_not_deletion(self):
        placeholder = self.write("Untitled.base")
        large = self.write("raw/papers/large.pdf")
        with large.open("wb") as stream:
            stream.truncate(5 * 1024 * 1024 + 1)
        errors, warnings = repository_hygiene(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)
        self.assertTrue(placeholder.exists() and large.exists())


if __name__ == "__main__":
    unittest.main()
