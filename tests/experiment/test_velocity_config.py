# std-lib imports

# 3 party imports
import pytest

# package imports
from trajectoryflow.experiment.config import load_benchmark_config


def test_load_velocity_config_and_tasks(tmp_path):
    path = tmp_path / "benchmark.toml"
    path.write_text(
        """
data_root = "data"
output_dir = "runs/test"
seeds = [0]

[evaluation]
sample_batch_size = 2

[runtime]
store_cache_size = 0

[evaluation.velocity]
enabled = true
reference_dir = "refs"
n_cells = 123
batch_size = 64
plots = false
plot_max_cells = 500
plot_max_arrows = 50
cell_id_column = "cell_id"
color_by = "cell_type"

[[models]]
name = "dummy"

[[splits]]
name = "velocity"
fit_timepoints = ["5h"]

[[splits.test_velocity_tasks]]
name = "pts_test"
reference = "pts"
timepoints = ["5h"]
partition = "all"
""".strip(),
        encoding="utf-8",
    )

    config = load_benchmark_config(path)

    assert config.evaluation.velocity.enabled
    assert config.evaluation.velocity.reference_dir.name == "refs"
    assert config.evaluation.sample_batch_size == 2
    assert config.runtime.store_cache_size == 0
    assert config.evaluation.velocity.n_cells == 123
    assert config.evaluation.velocity.batch_size == 64
    assert not config.evaluation.velocity.plots

    task = config.splits[0].test_velocity_tasks[0]
    assert task.task_name == "pts_test"
    assert task.reference == "pts"
    assert task.timepoints == ("5h",)
    assert task.partition == "all"


def test_config_rejects_unknown_options(tmp_path):
    path = tmp_path / "benchmark.toml"
    path.write_text(
        '''
data_root = "data"
save_predicitons = true

[[models]]
name = "dummy"

[[splits]]
name = "test"
fit_timepoints = ["5h"]

[[splits.test_velocity_tasks]]
reference = "pts"
timepoints = ["5h"]
'''.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown benchmark option"):
        load_benchmark_config(path)
