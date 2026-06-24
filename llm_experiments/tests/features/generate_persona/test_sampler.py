import pytest
from features.generate_persona.sampler import WeightedDistribution, PersonaSampler
from features.generate_persona.models import Persona


def test_weighted_distribution_sampling():
    """
    Test that WeightedDistribution samples elements correctly.
    """
    data = {"Option A": 0.8, "Option B": 0.2}
    dist = WeightedDistribution.from_dict(data)

    assert dist.labels == ("Option A", "Option B")
    assert dist.weights == (0.8, 0.2)

    for _ in range(50):
        val = dist.sample()
        assert val in ("Option A", "Option B")


def test_weighted_distribution_empty_raises():
    dist = WeightedDistribution((), ())
    with pytest.raises(ValueError, match="Cannot sample from an empty distribution"):
        dist.sample()


def test_persona_sampler():
    """
    Test PersonaSampler works with dummy WeightedDistributions.
    """
    race = WeightedDistribution.from_dict({"Parda": 1.0})
    gender = WeightedDistribution.from_dict({"Feminino": 1.0})
    age_group = WeightedDistribution.from_dict({"18-24": 1.0})
    region = WeightedDistribution.from_dict({"Nordeste": 1.0})
    urban_rural = WeightedDistribution.from_dict({"Urbana": 1.0})
    education = WeightedDistribution.from_dict({"Ensino Médio": 1.0})
    employment = WeightedDistribution.from_dict({"Ocupada": 1.0})
    income = WeightedDistribution.from_dict({"R$ 1.000 a R$ 2.000": 1.0})
    health_assessment = WeightedDistribution.from_dict({"Bom": 1.0})
    chronic_disease = WeightedDistribution.from_dict({"Não": 1.0})
    marital_status = WeightedDistribution.from_dict({"Solteiro(a)": 1.0})
    religion = WeightedDistribution.from_dict({"Católica": 1.0})
    inflation = 0.5

    sampler = PersonaSampler(
        race=race,
        gender=gender,
        age_group=age_group,
        region=region,
        urban_rural=urban_rural,
        education=education,
        employment=employment,
        income=income,
        health_assessment=health_assessment,
        chronic_disease=chronic_disease,
        marital_status=marital_status,
        religion=religion,
        inflation_rate=inflation,
    )

    persona = sampler.sample()
    assert isinstance(persona, Persona)
    assert persona.demographic.race == "Parda"
    assert persona.demographic.gender == "Feminino"
    assert persona.demographic.age_group == "18-24"
    assert persona.demographic.region == "Nordeste"
    assert persona.demographic.urban is True
    assert persona.economic.education_level == "Ensino Médio"
    assert persona.economic.employment_status == "Ocupada"
    assert persona.economic.income_per_capita == "R$ 1.000 a R$ 2.000"
    assert persona.economic.inflation_rate == 0.5
    assert persona.health.health_self_assessment == "Bom"
    assert persona.health.has_chronic_disease is False
    assert persona.social.marital_status == "Solteiro(a)"
    assert persona.social.religion == "Católica"

    batch = sampler.sample_batch(5)
    assert len(batch) == 5
    for p in batch:
        assert isinstance(p, Persona)
