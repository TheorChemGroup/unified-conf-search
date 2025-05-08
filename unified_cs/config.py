from pydantic.dataclasses import dataclass, Field
import yaml
from typing import Literal, Optional, Union


@dataclass
class BOConfig:
    # rolling_window_size: int = 5
    # rolling_std_threshold: float = 0.15
    # rolling_mean_threshold: float = 1.
    use_physics_kernel: bool = True
    acquisition_function: Literal["ei", "iv"] = "iv"


@dataclass
class RotamerAnalyzerConfig:
    bondlength_threshold: float
    valence_threshold: float
    fixed_dihedral_threshold: float
    stereocenter_rmsd_threshold: float


@dataclass
class InitialGeneratorConfig:
    mode: Literal["random", "mcr", "grid", "existing"] = Field(default="grid")
    # grid
    rotamer_analysis: Optional[RotamerAnalyzerConfig] = Field(default=None)
    cache_path: Optional[str] = None
    show_status: Optional[bool] = None
    # random
    num_iterations: Optional[int] = None
    # existing
    existing_path: Optional[str] = None

    def __post_init__(self):
        if self.mode == "existing" and not self.existing_path:
            raise ValueError(
                "existing_path must be provided when mode is 'existing'")
        if self.mode != "existing" and self.existing_path is not None:
            raise ValueError(
                "existing_path must be None unless mode is 'existing'")


@dataclass
class XtbLevel:
    flags: str = "--gfn2 --alpb water"
    force_constant: float = 5.0
    executable: str = 'xtb'
    _program_name: str = 'xtb'


@dataclass
class MmffLevel:
    variant: str = "MMFF94"
    _program_name: str = 'rdkit'


LevelOfTheoryConfig = Union[XtbLevel, MmffLevel]


@dataclass
class GlobalConfig:
    input_sdf: str
    max_iterations: int
    max_nothing_new: int
    ik_config: InitialGeneratorConfig
    bo_config: BOConfig
    leveloftheory: LevelOfTheoryConfig
    add_bonds: list[tuple[int, int]] = Field(default_factory=list)
    request_fixed: list[tuple[int, int]] = Field(default_factory=list)
    added_bonds: list[tuple[int, int]] = Field(default_factory=list)
    debug_mode: bool = False
    logfile: Optional[str] = None


def load_config(config_path: str) -> GlobalConfig:
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = GlobalConfig(**raw_config)
    return config
