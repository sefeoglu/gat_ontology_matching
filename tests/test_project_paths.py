import unittest

from src.project_paths import PROJECT_ROOT, load_config, get_dataset_dir, get_alignment_dir


class ProjectPathTest(unittest.TestCase):
    def test_project_root_is_repo_root(self):
        self.assertEqual(PROJECT_ROOT.name, "gat_ontology_matching")
        self.assertTrue((PROJECT_ROOT / "config.ini").exists())

    def test_config_and_dataset_paths_resolve_inside_repo(self):
        self.assertTrue(load_config().read("config.ini"))
        self.assertTrue(get_dataset_dir("conference").is_dir())
        self.assertTrue(get_alignment_dir("conference").is_dir())


if __name__ == "__main__":
    unittest.main()
