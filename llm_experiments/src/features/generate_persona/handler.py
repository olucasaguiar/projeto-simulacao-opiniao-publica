from settings import settings
from infrastructure.cache import DistributionCache
from infrastructure.sidra import SidraClient
from .sampler import WeightedDistribution, PersonaSampler
from .models import Persona
from .adapters import (
    censo_adapter,
    pnad_adapter,
    ipca_adapter,
    pns_adapter,
    civil_adapter,
)


class DistributionBuilder:
    def __init__(self, client: SidraClient, cache: DistributionCache):
        self._client = client
        self._cache = cache

    def build_sampler(self) -> PersonaSampler:
        race_dist = censo_adapter.get_race_distribution(self._client, self._cache)
        gender_dist = censo_adapter.get_gender_distribution(self._client, self._cache)
        region_dist = censo_adapter.get_region_distribution(self._client, self._cache)
        age_dist = censo_adapter.get_age_group_distribution(self._client, self._cache)
        urban_dist = censo_adapter.get_urban_rural_distribution(
            self._client, self._cache
        )
        religion_dist = censo_adapter.get_religion_distribution(
            self._client, self._cache
        )

        employment_dist = pnad_adapter.get_employment_distribution(
            self._client, self._cache
        )
        education_dist = pnad_adapter.get_education_distribution(
            self._client, self._cache
        )
        income_dist = pnad_adapter.get_income_distribution(self._client, self._cache)

        inflation_rate = ipca_adapter.get_monthly_inflation(self._client, self._cache)

        health_assessment_dist = pns_adapter.get_health_assessment_distribution(
            self._client, self._cache
        )
        chronic_disease_dist = pns_adapter.get_chronic_disease_distribution(
            self._client, self._cache
        )

        marital_status_dist = civil_adapter.get_marital_status_distribution(
            self._client, self._cache
        )

        return PersonaSampler(
            race=WeightedDistribution.from_dict(race_dist),
            gender=WeightedDistribution.from_dict(gender_dist),
            age_group=WeightedDistribution.from_dict(age_dist),
            region=WeightedDistribution.from_dict(region_dist),
            urban_rural=WeightedDistribution.from_dict(urban_dist),
            education=WeightedDistribution.from_dict(education_dist),
            employment=WeightedDistribution.from_dict(employment_dist),
            income=WeightedDistribution.from_dict(income_dist),
            health_assessment=WeightedDistribution.from_dict(health_assessment_dist),
            chronic_disease=WeightedDistribution.from_dict(chronic_disease_dist),
            marital_status=WeightedDistribution.from_dict(marital_status_dist),
            religion=WeightedDistribution.from_dict(religion_dist),
            inflation_rate=inflation_rate,
        )


def generate_batch(
    client: SidraClient,
    cache: DistributionCache,
    count: int = settings.persona_generation.default_amount,
) -> list[Persona]:
    builder = DistributionBuilder(client, cache)
    sampler = builder.build_sampler()
    return sampler.sample_batch(count)


def generate_one(client: SidraClient, cache: DistributionCache) -> Persona:
    builder = DistributionBuilder(client, cache)
    sampler = builder.build_sampler()
    return sampler.sample()
