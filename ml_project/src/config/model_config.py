from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    model_type: str = field(default='LogisticRegression')
    random_state: int = field(default=42)
    n_estimators: int = field(default=150)

