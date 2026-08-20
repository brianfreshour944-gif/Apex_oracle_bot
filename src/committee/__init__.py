"""Committee package for Apex Oracle Bot ensemble decision system."""

from .decision_transformer import (
    DecisionTransformer,
    get_decision_transformer,
    run_decision_transformer,
    train_decision_transformer,
    DTConfig,
    build_state_vector,
    build_action_vector,
    encode_regime,
    encode_brain_votes,
    encode_features,
)

from .bayesian_transformer import (
    BayesianTransformerBrain,
    EnsemblePrediction,
    get_bayesian_transformer,
    bayesian_transformer_brain,
    train_temperature_scaling,
    save_calibration,
    compute_ece,
)

from .batch_ensemble import (
    BatchEnsembleLinear,
    BatchEnsembleLayerNorm,
    BatchEnsembleDropout,
    BatchEnsembleTransformer,
    replace_linear_with_batchensemble,
    convert_model_to_batchensemble,
)

from src.ood_discriminator import (
    OODDiscriminator,
    get_ood_discriminator,
    reset_ood_discriminator,
    check_ood_and_override,
    build_ood_state_vector,
    TemporalEMASmoother,
)

__all__ = [
    "DecisionTransformer",
    "get_decision_transformer",
    "run_decision_transformer",
    "train_decision_transformer",
    "DTConfig",
    "build_state_vector",
    "build_action_vector",
    "encode_regime",
    "encode_brain_votes",
    "encode_features",
    "BayesianTransformerBrain",
    "EnsemblePrediction",
    "get_bayesian_transformer",
    "bayesian_transformer_brain",
    "train_temperature_scaling",
    "save_calibration",
    "compute_ece",
    "BatchEnsembleLinear",
    "BatchEnsembleLayerNorm",
    "BatchEnsembleDropout",
    "BatchEnsembleTransformer",
    "replace_linear_with_batchensemble",
    "convert_model_to_batchensemble",
    "OODDiscriminator",
    "get_ood_discriminator",
    "reset_ood_discriminator",
    "check_ood_and_override",
    "build_ood_state_vector",
    "TemporalEMASmoother",
]
