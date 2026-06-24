from pathlib import Path
from typing import List, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class LlmModelConfig(BaseModel):
    id: str
    provider: str
    adapter: str
    huggingface_id: Optional[str] = None


class MaritacaConfig(BaseModel):
    base_url: str
    timeout: float


class LocalModelConfig(BaseModel):
    device: str
    quantization: str


class LlmConfig(BaseModel):
    active_models: List[LlmModelConfig]
    maritaca: MaritacaConfig
    local: LocalModelConfig


class SidraApiConfig(BaseModel):
    base_url: str
    timeout: float


class SidraDefaultsConfig(BaseModel):
    all_sentinel: str


class SidraConfig(BaseModel):
    api: SidraApiConfig
    defaults: SidraDefaultsConfig


class CacheConfig(BaseModel):
    table_name: str
    default_source: str


class PersonaGenerationConfig(BaseModel):
    default_amount: int
    json_indent: int


class PathsConfig(BaseModel):
    distributions_db: str
    report_dir: str
    report_images_subdir: str
    report_filename: str


class ClassificationItem(BaseModel):
    id: int
    categories: List[str]


class QueryConfig(BaseModel):
    table_id: int
    period: str
    variable_id: str
    location_level: str
    classifications: List[ClassificationItem] = Field(default_factory=list)
    mappings: Dict[str, str] = Field(default_factory=dict)
    cache_key: str
    cache_ttl_days: int


class CensoConfig(BaseModel):
    race: QueryConfig
    gender: QueryConfig
    region: QueryConfig
    age_group: QueryConfig
    urban_rural: QueryConfig
    religion: QueryConfig


class CivilConfig(BaseModel):
    marital_status: QueryConfig


class IpcaConfig(BaseModel):
    inflation: QueryConfig


class PnadConfig(BaseModel):
    employment: QueryConfig
    education: QueryConfig
    income: QueryConfig


class PnsConfig(BaseModel):
    health_assessment: QueryConfig
    chronic_disease: QueryConfig
    scale_splits: Dict[str, float] = Field(default_factory=dict)


class ReportDataSource(BaseModel):
    name: str
    table_id: int
    period: str


class ReportConfig(BaseModel):
    data_sources: List[ReportDataSource]


class Settings(BaseModel):
    llm: LlmConfig
    sidra: SidraConfig
    cache: CacheConfig
    persona_generation: PersonaGenerationConfig
    paths: PathsConfig
    censo: CensoConfig
    civil: CivilConfig
    ipca: IpcaConfig
    pnad: PnadConfig
    pns: PnsConfig
    report: ReportConfig

    @classmethod
    def load_from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# Resolve path to config.yaml relative to project root
config_path = Path(__file__).parent.parent / "config.yaml"
settings = Settings.load_from_yaml(config_path)
