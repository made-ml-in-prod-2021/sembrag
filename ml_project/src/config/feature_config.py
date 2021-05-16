from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    numerical_features: List[str]
    target_col: str
    num_impute_strategy: str = field(default='median')
    cat_impute_strategy: str = field(default='most_frequent')
    categorical_features: List[str] = field(default_factory=list)
    features_to_drop: List[str] = field(default_factory=list)
    scale_features: bool = field(default=False)

