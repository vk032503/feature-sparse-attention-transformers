"""Feature-Sparse Attention Transformers package."""

from .feature_sparse_attention import (
    FeatureSparseAttention,
    FeatureSelector,
    VarianceFeatureSelector,
    GradientFeatureSelector,
    LearnedFeatureSelector,
)
from .transformer_integration import (
    replace_attention_layers,
    configure_layer_sparsity,
)
from .benchmark import run_benchmark, BenchmarkConfig
from .visualization import (
    plot_sparsity_patterns,
    plot_performance_comparison,
    plot_feature_importance,
)

__version__ = "0.1.0"
__all__ = [
    "FeatureSparseAttention",
    "FeatureSelector",
    "VarianceFeatureSelector",
    "GradientFeatureSelector",
    "LearnedFeatureSelector",
    "replace_attention_layers",
    "configure_layer_sparsity",
    "run_benchmark",
    "BenchmarkConfig",
    "plot_sparsity_patterns",
    "plot_performance_comparison",
    "plot_feature_importance",
]