from __future__ import annotations
import unittest
import torch
from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import training_core_final as core

# HELPER: Build the tiny model.
def _build_tiny_model() -> core.MultitaskLSTM:
    return core.MultitaskLSTM(
        categorical_vocab_sizes={"a": 10, "b": 5}, numerical_dim=3, offer_dim=2, next_activity_classes=7,
        hidden_size=16, num_layers=2, dropout=0.1, head_hidden_size=8,
    )

# HELPER: Build optimizer.
def _build_optimizer(config: baseline.BaselineRunConfig, model: core.MultitaskLSTM) -> torch.optim.Optimizer:
    head_prefixes = {
        "outcome": ("outcome_head.", float(config.outcome_lr_scale)),
        "next_activity": ("next_activity_head.", float(config.next_activity_lr_scale)),
        "remaining_time": ("remaining_time_head.", float(config.remaining_time_lr_scale)),
    }
    head_param_groups: dict[str, list[torch.nn.Parameter]] = {key: [] for key in head_prefixes}
    trunk_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        matched = next((key for key, (prefix, _) in head_prefixes.items() if name.startswith(prefix)), None)
        if matched is None:
            trunk_params.append(param)
        else:
            head_param_groups[matched].append(param)
    specs = [{"params": trunk_params, "lr": config.learning_rate, "group_name": "trunk"}]
    for key, (_, scale) in head_prefixes.items():
        specs.append({"params": head_param_groups[key], "lr": config.learning_rate * scale, "group_name": key})
    return torch.optim.AdamW(specs, weight_decay=config.weight_decay)

class PerHeadLearningRateTests(unittest.TestCase):
    # Verify shared optimizer helper builds four LR groups.
    def test_shared_optimizer_helper_builds_four_lr_groups(self) -> None:
        # Build the optimizer through the shared core helper expected by E_05 and E_06.
        model = _build_tiny_model()
        optimizer = core.build_multitask_optimizer(
            model, learning_rate=1e-3, weight_decay=1e-4, outcome_lr_scale=0.3, next_activity_lr_scale=1.0,
            remaining_time_lr_scale=1.5,
        )

        # Verify the four named groups and their group-specific base learning rates.
        by_name = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(by_name), {"trunk", "outcome", "next_activity", "remaining_time"})
        self.assertAlmostEqual(by_name["trunk"]["lr"], 1e-3)
        self.assertAlmostEqual(by_name["outcome"]["lr"], 1e-3 * 0.3)
        self.assertAlmostEqual(by_name["next_activity"]["lr"], 1e-3 * 1.0)
        self.assertAlmostEqual(by_name["remaining_time"]["lr"], 1e-3 * 1.5)

    # Verify four groups with expected LR products.
    def test_four_groups_with_expected_lr_products(self) -> None:
        config = baseline.BaselineRunConfig(
            learning_rate=1e-3, outcome_lr_scale=0.3, next_activity_lr_scale=1.0, remaining_time_lr_scale=1.5,
        )
        model = _build_tiny_model()
        optimizer = _build_optimizer(config, model)
        by_name = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(by_name), {"trunk", "outcome", "next_activity", "remaining_time"})
        self.assertAlmostEqual(by_name["trunk"]["lr"], 1e-3)
        self.assertAlmostEqual(by_name["outcome"]["lr"], 1e-3 * 0.3)
        self.assertAlmostEqual(by_name["next_activity"]["lr"], 1e-3 * 1.0)
        self.assertAlmostEqual(by_name["remaining_time"]["lr"], 1e-3 * 1.5)

    # Verify partition is disjoint and complete.
    def test_partition_is_disjoint_and_complete(self) -> None:
        config = baseline.BaselineRunConfig(learning_rate=1e-3)
        model = _build_tiny_model()
        optimizer = _build_optimizer(config, model)
        grouped_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
        model_ids = [id(param) for param in model.parameters()]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(model_ids))

    # Verify default scales keep heads at trunk LR.
    def test_default_scales_keep_heads_at_trunk_lr(self) -> None:
        config = baseline.BaselineRunConfig(learning_rate=2.5e-4)
        self.assertEqual(config.next_activity_lr_scale, 1.0)
        self.assertEqual(config.remaining_time_lr_scale, 1.0)
        model = _build_tiny_model()
        optimizer = _build_optimizer(config, model)
        by_name = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertAlmostEqual(by_name["next_activity"]["lr"], 2.5e-4)
        self.assertAlmostEqual(by_name["remaining_time"]["lr"], 2.5e-4)

class CosineScheduleTMaxTests(unittest.TestCase):
    # Verify decay completes at t max and holds the floor.
    def test_decay_completes_at_t_max_and_holds_the_floor(self) -> None:
        # Mirror the production loop, where the scheduler steps only while the epoch is within T_max.
        # The LR therefore reaches the floor at T_max and never climbs back up afterward.
        t_max = 5
        min_lr = 1e-6
        max_epochs = 8
        config = baseline.BaselineRunConfig(learning_rate=1e-3, lr_scheduler_t_max=t_max, lr_scheduler_min_lr=min_lr)
        model = _build_tiny_model()
        optimizer = _build_optimizer(config, model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.lr_scheduler_t_max, eta_min=config.lr_scheduler_min_lr,
        )
        trunk_lrs = []
        for epoch in range(1, max_epochs + 1):
            optimizer.step()  # mirror the real loop so the scheduler step order is valid
            trunk_lrs.append(float(optimizer.param_groups[0]["lr"]))
            if epoch <= config.lr_scheduler_t_max:
                scheduler.step()
        # The decay is strictly monotonic until the floor is reached at T_max.
        for earlier, later in zip(trunk_lrs[:t_max], trunk_lrs[1 : t_max + 1]): self.assertGreater(earlier, later)
        # Every epoch beyond T_max trains at the floor instead of a rising LR.
        for lr in trunk_lrs[t_max:]: self.assertAlmostEqual(lr, min_lr, places=9)

    # Verify default t max is decoupled from max epochs.
    def test_default_t_max_is_decoupled_from_max_epochs(self) -> None:
        config = baseline.BaselineRunConfig()
        self.assertEqual(config.lr_scheduler_t_max, 15)
        self.assertEqual(config.max_epochs, 40)

class PrefixBucketSummaryTests(unittest.TestCase):
    # Verify summary rows, skip empty buckets and keep columns.
    def test_summary_rows_skip_empty_buckets_and_keep_columns(self) -> None:
        bucket_metrics = {
            "1": {"n_prefixes": 0},
            "2-5": {
                "n_prefixes": 10,
                "outcome_accuracy": 0.5,
                "outcome_macro_f1": 0.4,
                "outcome_weighted_f1": 0.45,
                "outcome_balanced_accuracy": 0.42,
                "remaining_time_mae_seconds": 1234.0,
            },
        }
        rows = baseline.prefix_bucket_summary_rows(bucket_metrics)
        self.assertEqual([row["prefix_bucket"] for row in rows], ["2-5"])
        self.assertEqual(
            set(rows[0]),
            {
                "prefix_bucket",
                "n_prefixes",
                "outcome_accuracy",
                "outcome_macro_f1",
                "outcome_weighted_f1",
                "outcome_balanced_accuracy",
                "remaining_time_mae_seconds",
            },
        )

if __name__ == "__main__":
    unittest.main()