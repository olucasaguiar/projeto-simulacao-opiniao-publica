from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class DemographicProfile(BaseModel):
    age_group: str  # "18-24", "25-34", ... (PNAD C58 age brackets)
    gender: str  # "Masculino", "Feminino"
    race: str  # "Parda", "Branca", ... (Censo 9605 C86)
    region: str  # "Sudeste", "Nordeste", ... (Censo 9605 N2)
    urban: bool  # True/False (Censo 9923)


class EconomicProfile(BaseModel):
    education_level: (
        str  # "Sem instrução", "Fundamental incompleto", ... (PNAD 6461 C11255)
    )
    employment_status: str  # "Ocupada", "Desocupada", "Fora da força" (PNAD 6318)
    income_per_capita: str  # Bracket derived from PNAD 6381
    inflation_rate: float  # Latest monthly IPCA (7060 V63)


class HealthProfile(BaseModel):
    health_self_assessment: (
        str  # "Muito bom", "Bom", "Regular", "Ruim", "Muito ruim" (PNS 8151)
    )
    has_chronic_disease: bool  # Probability-based sampling (PNS 8168)


class SocialProfile(BaseModel):
    marital_status: (
        str  # "Solteiro(a)", "Casado(a)", "Divorciado(a)", "Viúvo(a)" (RC 5565)
    )
    religion: str  # "Católica", "Evangélica", "Espírita", ... (Censo 9551)


class Persona(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    demographic: DemographicProfile
    economic: EconomicProfile
    health: HealthProfile
    social: SocialProfile
