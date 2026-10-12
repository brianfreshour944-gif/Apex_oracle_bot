"""Tests for PBT cycle honesty (src/population_trainer.py).

The live PBT cycle passed an empty `live_components` and then logged
"PBT: Applied best config from worker -1 to live components" and a
"PBT cycle completed: pop=0" line every run. Neither was true: there is no
live reward feed, so the population is never initialized and nothing is
applied. `apply_best_to_live` now reports whether it actually mutated a
component, and the caller skips the fake completion log when there is no
population.
"""
from types import SimpleNamespace

from src.population_trainer import PopulationTrainer


def _trainer(tmp_path) -> PopulationTrainer:
    # Point at a path that does not exist so a previous run's state can't leak
    # into the test population.
    return PopulationTrainer(population_size=4, state_path=str(tmp_path / "pbt_state.json"))


def test_population_stats_empty_until_initialized(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer.get_population_stats() == {}


def test_apply_best_to_live_with_no_components_is_a_noop(tmp_path, capsys):
    trainer = _trainer(tmp_path)
    applied = trainer.apply_best_to_live({})
    assert applied is False
    # The logger writes to stdout; the misleading line must not appear.
    assert "Applied best config" not in capsys.readouterr().out


def test_apply_best_to_live_reports_true_when_it_mutates_a_component(tmp_path, capsys):
    trainer = _trainer(tmp_path)
    meta_learner = SimpleNamespace(learning_rate=0.0, min_weight=0.0, max_weight=0.0)
    applied = trainer.apply_best_to_live({"meta_learner": meta_learner})
    assert applied is True
    assert "Applied best config" in capsys.readouterr().out
    assert meta_learner.learning_rate != 0.0
