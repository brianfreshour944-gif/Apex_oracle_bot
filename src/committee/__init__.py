"""Committee package for Apex Oracle Bot ensemble decision system.

Imports are LAZY (PEP 562 __getattr__). The brain modules pull in torch and
friends at import time; eager re-exports here meant that importing *any*
committee submodule (even the pure-Python models.py) required torch to be
installed. With lazy resolution, `from src.committee.models import BrainVote`
works torch-free, and `from src.committee import DecisionTransformer` still
resolves exactly as before -- torch is only imported when actually used.
"""

import importlib

_SUBMODULE_EXPORTS = {
    # batch_ensemble
    "BatchEnsembleDropout": "batch_ensemble",
    "BatchEnsembleLayerNorm": "batch_ensemble",
    "BatchEnsembleLinear": "batch_ensemble",
    "BatchEnsembleTransformer": "batch_ensemble",
    "convert_model_to_batchensemble": "batch_ensemble",
    "replace_linear_with_batchensemble": "batch_ensemble",
    # bayesian_transformer
    "BayesianTransformerBrain": "bayesian_transformer",
    "EnsemblePrediction": "bayesian_transformer",
    "bayesian_transformer_brain": "bayesian_transformer",
    "compute_ece": "bayesian_transformer",
    "get_bayesian_transformer": "bayesian_transformer",
    "save_calibration": "bayesian_transformer",
    "train_temperature_scaling": "bayesian_transformer",
    # decision_gate
    "GateResult": "decision_gate",
    "check_decision_source_gate": "decision_gate",
    "get_gate_status_summary": "decision_gate",
    "log_gate_status": "decision_gate",
    # decision_transformer
    "DecisionTransformer": "decision_transformer",
    "DTConfig": "decision_transformer",
    "build_action_vector": "decision_transformer",
    "build_state_vector": "decision_transformer",
    "encode_brain_votes": "decision_transformer",
    "encode_features": "decision_transformer",
    "encode_regime": "decision_transformer",
    "get_decision_transformer": "decision_transformer",
    "run_decision_transformer": "decision_transformer",
    "train_decision_transformer": "decision_transformer",
    # hierarchical_skills
    "HierarchicalSkills": "hierarchical_skills",
    "OptionsCriticTrainer": "hierarchical_skills",
    "SkillConfig": "hierarchical_skills",
    "SkillCritic": "hierarchical_skills",
    "SkillEncoder": "hierarchical_skills",
    "SkillLSTMExecutor": "hierarchical_skills",
    "TerminationHead": "hierarchical_skills",
    "get_hierarchical_skills": "hierarchical_skills",
    "get_skills_trainer": "hierarchical_skills",
    "reset_hierarchical_skills": "hierarchical_skills",
    "run_hierarchical_skills": "hierarchical_skills",
    # ood_discriminator
    "OODDiscriminator": "ood_discriminator",
    "TemporalEMASmoother": "ood_discriminator",
    "build_ood_state_vector": "ood_discriminator",
    "check_ood_and_override": "ood_discriminator",
    "get_ood_discriminator": "ood_discriminator",
    "reset_ood_discriminator": "ood_discriminator",
}

__all__ = list(_SUBMODULE_EXPORTS.keys())


def __getattr__(name: str):
    module_name = _SUBMODULE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    # Cache on the module so subsequent lookups skip the import machinery.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
