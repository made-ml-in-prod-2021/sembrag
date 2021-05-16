from dataclasses import dataclass, field


@dataclass()
class DataFileParams:
    path : str
    sep : str = field(default=',')


@dataclass()
class SplittingParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=42)
