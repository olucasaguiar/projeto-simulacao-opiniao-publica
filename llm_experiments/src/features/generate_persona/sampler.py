import random
from dataclasses import dataclass
from .models import (
    DemographicProfile,
    EconomicProfile,
    HealthProfile,
    SocialProfile,
    Persona,
)


@dataclass(frozen=True)
class WeightedDistribution:
    labels: tuple[str, ...]
    weights: tuple[float, ...]

    def sample(self) -> str:
        if not self.labels:
            raise ValueError("Cannot sample from an empty distribution")
        return random.choices(self.labels, weights=self.weights, k=1)[0]

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "WeightedDistribution":
        return cls(labels=tuple(data.keys()), weights=tuple(data.values()))


class PersonaSampler:
    def __init__(
        self,
        race: WeightedDistribution,
        gender: WeightedDistribution,
        age_group: WeightedDistribution,
        region: WeightedDistribution,
        urban_rural: WeightedDistribution,
        education: WeightedDistribution,
        employment: WeightedDistribution,
        income: WeightedDistribution,
        health_assessment: WeightedDistribution,
        chronic_disease: WeightedDistribution,
        marital_status: WeightedDistribution,
        religion: WeightedDistribution,
        inflation_rate: float,
    ):
        self._race = race
        self._gender = gender
        self._age_group = age_group
        self._region = region
        self._urban_rural = urban_rural
        self._education = education
        self._employment = employment
        self._income = income
        self._health_assessment = health_assessment
        self._chronic_disease = chronic_disease
        self._marital_status = marital_status
        self._religion = religion
        self._inflation_rate = inflation_rate

    def sample(self) -> Persona:
        urban_label = self._urban_rural.sample()
        # Handle different representations of urban/rural
        is_urban = urban_label in ("Urbana", "Urbano", "true", "True", True)

        chronic_label = self._chronic_disease.sample()
        has_chronic = chronic_label in ("Sim", "sim", "true", "True", True)

        return Persona(
            demographic=DemographicProfile(
                age_group=self._age_group.sample(),
                gender=self._gender.sample(),
                race=self._race.sample(),
                region=self._region.sample(),
                urban=is_urban,
            ),
            economic=EconomicProfile(
                education_level=self._education.sample(),
                employment_status=self._employment.sample(),
                income_per_capita=self._income.sample(),
                inflation_rate=self._inflation_rate,
            ),
            health=HealthProfile(
                health_self_assessment=self._health_assessment.sample(),
                has_chronic_disease=has_chronic,
            ),
            social=SocialProfile(
                marital_status=self._marital_status.sample(),
                religion=self._religion.sample(),
            ),
        )

    def sample_batch(self, count: int) -> list[Persona]:
        return [self.sample() for _ in range(count)]
