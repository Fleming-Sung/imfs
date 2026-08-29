import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import mujoco

from deploy.course import build_scene


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_XML_SHA256 = "ad4c6af61c278903bd89d9a17206e1c46eb9b93792c9a8fe34dd539cb971edea"
JOINTS = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
]


class OfficialModelTest(unittest.TestCase):
    def test_untouched_official_source_and_complete_ankle_links(self):
        official = ROOT / "assets" / "SF_TRON1A" / "xml" / "robot_official.xml"
        self.assertEqual(hashlib.sha256(official.read_bytes()).hexdigest(), OFFICIAL_XML_SHA256)
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "assets" / "SF_TRON1A" / "xml" / "robot_deploy.xml"))
        for side in ("L", "R"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}_Link")
            geom_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                for geom_id in range(model.ngeom) if model.geom_bodyid[geom_id] == body_id
            ]
            # The unnamed geom is the official visual mesh; the named one is its collision mesh.
            self.assertIn(None, geom_names)
            self.assertIn(f"ankle_{side}_collision", geom_names)
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot_{side}_site"), 0)
        self.assertEqual(
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)],
            JOINTS)
        self.assertAlmostEqual(model.opt.timestep, 0.001)

    def test_generated_course_scene_compiles(self):
        cfg = json.loads((ROOT / "config" / "plum_piles.json").read_text())
        train_cfg = json.loads((ROOT / "checkpoints" / "training_config.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.xml"
            layout = build_scene(
                ROOT / "assets" / "SF_TRON1A" / "xml" / "robot_deploy.xml",
                path, cfg, train_cfg["init"]["reset_joint_angles"])
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(
                sum(point.support == "pile" for point in layout.footholds), cfg["num_piles"])
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_SITE, layout.footholds[-1].site_name), 0)


if __name__ == "__main__":
    unittest.main()
