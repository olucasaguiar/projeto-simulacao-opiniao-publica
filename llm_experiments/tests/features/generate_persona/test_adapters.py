from features.generate_persona.adapters import (
    censo_adapter,
    civil_adapter,
    ipca_adapter,
    pnad_adapter,
    pns_adapter,
)


def test_censo_adapter_race(sidra_client, db_cache):
    dist = censo_adapter.get_race_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {
        "Branca",
        "Preta",
        "Amarela",
        "Parda",
        "Indígena",
        "Sem declaração",
    }
    for label in dist.keys():
        assert label in expected_categories


def test_censo_adapter_gender(sidra_client, db_cache):
    dist = censo_adapter.get_gender_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {"Masculino", "Feminino"}
    for label in dist.keys():
        assert label in expected_categories
    assert "Masculino" in dist
    assert "Feminino" in dist


def test_censo_adapter_region(sidra_client, db_cache):
    dist = censo_adapter.get_region_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {"Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"}
    for label in dist.keys():
        assert label in expected_categories


def test_censo_adapter_age_group(sidra_client, db_cache):
    dist = censo_adapter.get_age_group_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    for label in dist.keys():
        assert isinstance(label, str)
        assert len(label) > 0


def test_censo_adapter_urban_rural(sidra_client, db_cache):
    dist = censo_adapter.get_urban_rural_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {"Urbana", "Rural"}
    for label in dist.keys():
        assert label in expected_categories


def test_censo_adapter_religion(sidra_client, db_cache):
    dist = censo_adapter.get_religion_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    for label in dist.keys():
        assert isinstance(label, str)


def test_civil_adapter_marital_status(sidra_client, db_cache):
    dist = civil_adapter.get_marital_status_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {
        "Solteiro(a)",
        "Casado(a)",
        "Divorciado(a)",
        "Viúvo(a)",
        "Desquitado(a) ou separado(a) judicialmente",
    }
    for label in dist.keys():
        assert (
            label in expected_categories
            or "solteir" in label.lower()
            or "casad" in label.lower()
        )


def test_ipca_adapter_inflation(sidra_client, db_cache):
    rate = ipca_adapter.get_monthly_inflation(sidra_client, db_cache)
    assert isinstance(rate, float)
    assert -20.0 < rate < 50.0


def test_pnad_adapter_employment(sidra_client, db_cache):
    dist = pnad_adapter.get_employment_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {"Ocupada", "Desocupada", "Fora da força"}
    for label in dist.keys():
        assert label in expected_categories
    assert "Ocupada" in dist


def test_pnad_adapter_education(sidra_client, db_cache):
    dist = pnad_adapter.get_education_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    for label in dist.keys():
        assert isinstance(label, str)
        assert len(label) > 0


def test_pnad_adapter_income(sidra_client, db_cache):
    dist = pnad_adapter.get_income_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    for label in dist.keys():
        assert isinstance(label, str)
        assert len(label) > 0


def test_pns_adapter_health_assessment(sidra_client, db_cache):
    dist = pns_adapter.get_health_assessment_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {
        "Muito bom",
        "Bom",
        "Regular",
        "Ruim",
        "Muito ruim",
        "Muito boa",
        "Boa",
    }
    for label in dist.keys():
        assert (
            label in expected_categories
            or "bom" in label.lower()
            or "ruim" in label.lower()
            or "regular" in label.lower()
        )


def test_pns_adapter_chronic_disease(sidra_client, db_cache):
    dist = pns_adapter.get_chronic_disease_distribution(sidra_client, db_cache)
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 100.0) < 1.0

    expected_categories = {"Sim", "Não"}
    for label in dist.keys():
        assert label in expected_categories
