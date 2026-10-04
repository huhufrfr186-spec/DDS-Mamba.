import unittest
import tempfile
from pathlib import Path
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from dds_mamba.config import Config
from dds_mamba.geometry import Crop, iou, xywh_to_center, center_to_xywh, region
from dds_mamba.mamba import bounded_gate, _BoundedProjection, ReferenceMamba
from dds_mamba.model import Network
from dds_mamba.encoders import TinyEncoders
from dds_mamba.controller import Controller, Candidate, map_evidence
from dds_mamba.memory import Memory
from dds_mamba.kalman import Kalman
from dds_mamba.losses import objective, focal_loss, teacher_probability, learning_rate
from dds_mamba.data import lasot_split, prepare, load_manifest
from dds_mamba.evaluation import bootstrap
from dds_mamba.training import negative_crop
from PIL import Image


def small_config(**kwargs):
    return Config(d_model=32, d_state=4, dt_rank=2, checkpoint_branches=False, **kwargs)


def make_state(cfg=None):
    cfg = cfg or small_config()
    identity = F.normalize(torch.ones(1, 384), dim=-1)
    app = F.normalize(torch.arange(1, 33).float()[None], dim=-1)
    return Controller(cfg, [100, 80, 30, 20], torch.ones(1, 32), app, identity, 640, 480)


def record(state, box=None, map_quality=.8, peak=.9, identity=.9, agreement=1.0, index=0, appearance=None):
    box = np.asarray(state.prediction if box is None else box, dtype=np.float64)
    cfg = state.cfg
    proposal = torch.ones(1, 32, requires_grad=True) if appearance is None else appearance
    output = {"position": torch.ones(1, 32) * 2, "appearance": proposal,
              "box": torch.tensor([[.5, .5, .2, .2]]), "logits": torch.zeros(1, 256)}
    return Candidate(output, Crop.around(box, 4), index, box, state.initial_identity.clone(),
                     torch.tensor([map_quality]), torch.tensor([peak]), torch.tensor([identity]),
                     torch.tensor([agreement]), torch.tensor([min(cfg.rho_max, map_quality * agreement)]))


class GeometryTests(unittest.TestCase):
    def test_box_crop_round_trip(self):
        box = np.array([121., 56., 24., 10.])
        np.testing.assert_allclose(xywh_to_center(center_to_xywh(box)), box)
        crop = Crop(100, 80, 160)
        np.testing.assert_allclose(crop.to_image(crop.to_crop(box)), box)
        self.assertAlmostEqual(iou(box, box), 1)
        self.assertFalse(crop.contains([500, 500, 10, 10]))

    def test_crop_pixel_centers(self):
        image = torch.arange(12).float().reshape(1, 1, 3, 4)
        result = region(image, [2, 1.5, 4, 3], 4)
        self.assertEqual(result.shape, (1, 1, 4, 4))
        # Square 4x4 identity sampling must preserve source centers exactly.
        square = torch.arange(16).float().reshape(1, 1, 4, 4)
        torch.testing.assert_close(region(square, [2, 2, 4, 4], 4, antialias=False), square)


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_projection_bounds_sum_and_correct_gradient(self):
        cfg = small_config()
        gate = bounded_gate(torch.randn(3, 256) * 20, cfg)
        self.assertTrue((gate >= .5).all() and (gate <= 1.5).all())
        torch.testing.assert_close(gate.sum(-1), torch.full((3,), 256.), atol=2e-5, rtol=0)
        value = torch.tensor([[.0, .7, 1.1, 2.2]], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: _BoundedProjection.apply(x, .5, 1.5), (value,), eps=1e-6, atol=1e-5))
        self.assertTrue(torch.autograd.gradcheck(lambda x: bounded_gate(x, cfg), (torch.randn(1, 8, dtype=torch.double, requires_grad=True),), atol=1e-5))

    def test_projection_rejects_infeasible_bounds(self):
        with self.assertRaises(ValueError):
            small_config(gate_min=1.1)

    def test_mamba_causality_and_cache_reset(self):
        mixer = ReferenceMamba(16, d_state=4, dt_rank=2)
        x = torch.randn(1, 9, 16)
        changed = x.clone()
        changed[:, 6:] += 100
        torch.testing.assert_close(mixer(x)[:, :6], mixer(changed)[:, :6])
        baseline = mixer(x)
        mixer(torch.randn_like(x))
        torch.testing.assert_close(mixer(x), baseline)
        dt = F.softplus(mixer.dt_proj.bias)
        self.assertTrue(((dt >= .001) & (dt <= .1)).all())

    def test_mamba_scan_against_independent_unroll(self):
        m = ReferenceMamba(8, d_state=3, dt_rank=2).double()
        x = torch.randn(1, 5, 8, dtype=torch.double)
        u, z = m.in_proj(x).chunk(2, -1)
        u = F.silu(m.conv1d(u.transpose(1, 2))[:, :, :5].transpose(1, 2))
        dt, b, c = m.x_proj(u).split([2, 3, 3], -1)
        dt = F.softplus(m.dt_proj(dt))
        # Separate scalar loops over channel and state dimensions.
        expected = torch.zeros_like(u)
        for channel in range(m.d_inner):
            state = torch.zeros(3, dtype=torch.double)
            for t in range(5):
                state = torch.exp(-dt[0, t, channel] * torch.exp(m.A_log[channel])) * state + dt[0, t, channel] * b[0, t] * u[0, t, channel]
                expected[0, t, channel] = (state * c[0, t]).sum() + m.D[channel] * u[0, t, channel]
        expected = m.out_proj(expected * F.silu(z))
        torch.testing.assert_close(m(x), expected)

    def test_token_order_linear_context_offset_and_gradients(self):
        model = Network(small_config(), TinyEncoders()).eval()
        args = [torch.randn(1, 64, 768), torch.randn(1, 256, 768), torch.tensor([[.5, .5, .2, .2]]),
                torch.randn(1, 32), torch.randn(1, 32), torch.randn(1, 384)]
        seen = {}
        hooks = [model.position.register_forward_pre_hook(lambda _, x: seen.update(position=x[0].shape)),
                 model.appearance.register_forward_pre_hook(lambda _, x: seen.update(appearance=x[0].shape))]
        out = model(*args)
        self.assertEqual(seen, {"position": torch.Size([1, 2, 32]), "appearance": torch.Size([1, 258, 32])})
        changed_args = args[:-1] + [args[-1] + 50]
        other = model(*changed_args)
        torch.testing.assert_close(out["spatial_outputs"], other["spatial_outputs"])
        offset = other["logits"] - out["logits"]
        torch.testing.assert_close(offset, offset[:, :1].expand_as(offset), atol=1e-5, rtol=1e-5)
        out["logits"].sum().backward()
        self.assertIsNotNone(model.appearance.blocks[0].mixer.x_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.encoders.parameters()))
        for hook in hooks:
            hook.remove()

    def test_map_concentration(self):
        quality, peak, probability = map_evidence(torch.zeros(1, 256))
        self.assertAlmostEqual(float(quality), 0, places=6)
        self.assertAlmostEqual(float(peak), .5)
        logits = torch.full((1, 256), -20.)
        logits[:, 0] = 20
        quality, _, _ = map_evidence(logits)
        self.assertGreater(float(quality), .99)


class ControllerTests(unittest.TestCase):
    def test_active_qacu_and_memory(self):
        state = make_state()
        old = state.appearance.clone()
        state.begin()
        candidate = record(state)
        output = state.finish([candidate])
        self.assertIsNotNone(output)
        torch.testing.assert_close(state.appearance, F.normalize(.2 * old + .8 * candidate.output["appearance"], dim=-1))
        self.assertAlmostEqual(state.reliability, .98, places=6)
        self.assertEqual(state.commit_count, 1)
        self.assertEqual(len(state.memory.entries), 1)
        self.assertEqual(state.memory.entries[0].age, 0)

    def test_weak_transition_output_uses_incoming_mode(self):
        state = make_state()
        initial_app = state.appearance
        initial_last = state.last_box.copy()
        for _ in range(3):
            state.begin()
            self.assertIsNotNone(state.finish([]))
        self.assertEqual(state.mode, "lost")
        self.assertEqual(state.incoming_mode, "active")
        self.assertIs(state.appearance, initial_app)
        np.testing.assert_allclose(state.last_box, initial_last)
        state.begin()
        self.assertIsNone(state.finish([]))

    def test_recovery_preserves_appearance_statistics_and_memory_order(self):
        state = make_state()
        state.mode = "lost"
        state.memory.write(state.initial_identity, .8)
        old_app = state.appearance
        state.reliability, state.last_rate, state.commit_count = .83, .61, 10
        state.begin()
        self.assertIsNone(state.finish([record(state, agreement=0)]))
        state.begin()
        candidate = record(state, agreement=0)
        self.assertIsNotNone(state.finish([candidate]))
        self.assertTrue(state.recovered)
        self.assertEqual(state.mode, "active")
        self.assertIs(state.appearance, old_app)
        self.assertEqual((state.reliability, state.last_rate, state.commit_count), (.83, .61, 10))
        self.assertEqual(len(state.memory.entries), 1)
        self.assertEqual(state.memory.entries[0].age, 2)

    def test_cache_needs_consecutive_consistent_records(self):
        state = make_state()
        state.mode = "lost"
        state.begin(); state.finish([record(state)])
        state.begin(); state.finish([])
        state.begin(); self.assertIsNone(state.finish([record(state)]))
        state.begin(); self.assertIsNotNone(state.finish([record(state)]))

    def test_innovation_rejects_without_changing_posterior(self):
        cfg = small_config()
        kf = Kalman([100, 80, 30, 20], 640, 480, cfg)
        kf.predict()
        x, p = kf.x.copy(), kf.P.copy()
        self.assertIsNone(kf.update([1e8, 1e8, 30, 20]))
        np.testing.assert_array_equal(kf.x, x)
        np.testing.assert_array_equal(kf.P, p)
        self.assertIsNotNone(kf.update([101, 80, 30, 20]))
        self.assertTrue((np.linalg.eigvalsh(kf.P) > 0).all())
        np.testing.assert_array_equal(np.diag(kf.Q), cfg.process_diagonal)
        np.testing.assert_array_equal(np.diag(kf.R), cfg.measurement_diagonal)

    def test_gradient_crosses_accepted_qacu_steps(self):
        state = make_state()
        first = torch.randn(1, 32, requires_grad=True)
        state.begin(); state.finish([record(state, appearance=first)])
        second = torch.randn(1, 32, requires_grad=True)
        state.begin(); state.finish([record(state, appearance=second)])
        state.appearance[:, 0].sum().backward()
        self.assertGreater(float(first.grad.abs().sum()), 0)
        self.assertGreater(float(second.grad.abs().sum()), 0)

    def test_raw_fallback_does_not_bypass_gates(self):
        state = make_state()
        state.mode = "lost"
        state.begin()
        a = record(state, map_quality=.1, index=0)
        b = record(state, map_quality=.2, index=1)
        self.assertIs(state.selected([a, b], raw_fallback=True), b)
        self.assertIsNone(state.finish([a, b]))
        self.assertEqual(len(state.cache), 0)

    def test_cache_rejects_inconsistent_identity_and_location(self):
        state = make_state()
        state.mode = "lost"
        state.begin(); state.finish([record(state)])
        state.begin()
        changed = record(state, box=[500, 400, 30, 20])
        self.assertIsNone(state.finish([changed]))
        self.assertEqual(len(state.cache), 1)
        state.begin()
        different_identity = record(state, box=[500, 400, 30, 20])
        different_identity.embedding = -state.initial_identity
        self.assertIsNone(state.finish([different_identity]))
        self.assertEqual(len(state.cache), 1)


class MemoryLossTests(unittest.TestCase):
    def test_memory_replace_lowest_utility_and_read_weights(self):
        cfg = small_config(memory_capacity=2)
        memory = Memory(cfg)
        a = F.normalize(torch.randn(1, 384), dim=-1)
        b = F.normalize(torch.randn(1, 384), dim=-1)
        memory.write(a, .71); memory.write(b, .95)
        memory.age()
        memory.write(a, .85)
        self.assertAlmostEqual(memory.entries[0].weight, .85)
        self.assertEqual(memory.entries[1].age, 1)
        model = Network(cfg, TinyEncoders())
        app, initial = torch.randn(1, 32), torch.randn(1, 384)
        keys, utilities = memory.tensors(app)
        query = F.normalize(model.memory_query(torch.cat([app, initial], -1)), dim=-1)
        scores = query @ keys.T * utilities
        expected = scores.softmax(-1) @ keys
        torch.testing.assert_close(model.read_memory(app, initial, memory), expected)
        self.assertFalse(memory.write(a, .69))

    def test_losses_masking_focal_formula_alignment(self):
        model = Network(small_config(), TinyEncoders())
        logits = torch.randn(1, 256, requires_grad=True)
        pos, app = torch.randn(1, 32), torch.randn(1, 32)
        output = {"box": torch.tensor([[.7, .4, .2, .2]], requires_grad=True), "logits": logits,
                  "position": pos, "appearance": app, "identity_projection": torch.randn(1, 384)}
        gt, identity = torch.tensor([[.5, .5, .2, .2]]), torch.randn(1, 384)
        loss, terms = objective(output, gt, identity, False, torch.randn(1, 32), True, model)
        self.assertTrue(all(float(v.detach()) == 0 for k, v in terms.items() if k != "ctr"))
        torch.testing.assert_close(loss, focal_loss(logits, torch.zeros_like(logits)))
        _, terms = objective(output, gt, identity, True, torch.randn(1, 32), False, model)
        self.assertEqual(float(terms["temp"].detach()), 0)
        self.assertGreater(float(terms["align"].detach()), 0)
        terms["align"].backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        p = torch.tensor([[.2, .7]])
        y = torch.tensor([[.3, .8]])
        logit = torch.logit(p)
        expected = -(y * (1-p)**2 * p.log() + (1-y) * p**2 * (1-p).log()).mean()
        torch.testing.assert_close(focal_loss(logit, y), expected)

    def test_teacher_and_lr_schedule(self):
        cfg = small_config()
        self.assertAlmostEqual(teacher_probability(0, cfg), 1)
        self.assertAlmostEqual(teacher_probability(10, cfg), .5)
        self.assertEqual(teacher_probability(20, cfg), 0)
        self.assertAlmostEqual(learning_rate(5, cfg), 2e-4)
        self.assertAlmostEqual(learning_rate(40, cfg), 0)

    def test_split_no_leakage_and_bootstrap_pairing(self):
        ids = [f"seq-{i}" for i in range(1120)]
        train, dev = lasot_split(ids)
        self.assertEqual((len(train), len(dev)), (896, 224))
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(lasot_split(ids[::-1]), (train, dev))
        with self.assertRaises(ValueError):
            lasot_split(ids, test_names=[ids[0]])
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            a.write_text("sequence,auc\na,0.6\nb,0.8\n")
            b.write_text("sequence,auc\na,0.5\nb,0.7\n")
            report = bootstrap([a], "auc", [b], resamples=50)
            self.assertAlmostEqual(report["mean"], .1)
            self.assertAlmostEqual(report["ci95"][0], .1)

    def test_standard_lasot_name_flags_and_inference_label_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "car" / "car-17"
            (directory / "img").mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (64, 64)).save(directory / "img" / f"{i+1:08d}.png")
            (directory / "groundtruth.txt").write_text("10,10,20,20\n11,10,20,20\n12,10,20,20\n")
            (directory / "full_occlusion.txt").write_text("0,1,0")
            (directory / "out_of_view.txt").write_text("0,0,1")
            manifest = prepare(root, "lasot", "dev", ["car-17"])
            self.assertEqual(manifest["sequences"][0]["name"], "car-17")
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            sequences, _ = load_manifest(path)
            boxes, visible = sequences[0].labels()
            np.testing.assert_array_equal(visible, [True, False, False])
            self.assertEqual(boxes.shape, (3, 4))
            # Moving/removing later labels cannot alter an inference manifest load.
            (directory / "groundtruth.txt").unlink()
            self.assertEqual(len(load_manifest(path)[0][0].frames), 3)

    def test_negative_crop_does_not_include_target(self):
        target = [90, 56, 16, 12]
        crop = negative_crop(target, 110, 192, 128)
        self.assertIsNotNone(crop)
        self.assertEqual(iou([crop.cx, crop.cy, crop.side, crop.side], target), 0)
        self.assertIsNone(negative_crop([96,64,192,128], 300, 192, 128))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
