from settings import settings
from infrastructure.sidra import SidraQueryBuilder, SidraVariableResponse


def test_censo_race_query(sidra_client):
    """
    Test real Censo 2022 query (Table 9606) for race/color distribution.
    """
    cfg = settings.censo.race
    builder = (
        SidraQueryBuilder(cfg.table_id)
        .select_periods(cfg.period)
        .select_variables(cfg.variable_id)
        .select_locations(cfg.location_level)
    )
    for c in cfg.classifications:
        builder.select_classification(c.id, c.categories)

    data = sidra_client.get_data(builder)

    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert isinstance(item, SidraVariableResponse)
        assert item.resultados
        for res in item.resultados:
            assert res.series


def test_civil_marital_status_query(sidra_client):
    """
    Test real Registro Civil query (Table 1206) for marital status.
    """
    cfg = settings.civil.marital_status
    builder = (
        SidraQueryBuilder(cfg.table_id)
        .select_periods(cfg.period)
        .select_variables(cfg.variable_id)
        .select_locations(cfg.location_level)
    )
    for c in cfg.classifications:
        builder.select_classification(c.id, c.categories)

    data = sidra_client.get_data(builder)

    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert isinstance(item, SidraVariableResponse)
        assert item.resultados


def test_ipca_inflation_query(sidra_client):
    """
    Test real IPCA query (Table 7060) for inflation rate.
    """
    cfg = settings.ipca.inflation
    builder = (
        SidraQueryBuilder(cfg.table_id)
        .select_periods(cfg.period)
        .select_variables(cfg.variable_id)
        .select_locations(cfg.location_level)
    )
    for c in cfg.classifications:
        builder.select_classification(c.id, c.categories)

    data = sidra_client.get_data(builder)

    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert isinstance(item, SidraVariableResponse)
        assert item.resultados


def test_pnad_employment_query(sidra_client):
    """
    Test real PNAD query (Table 6318) for employment status.
    """
    cfg = settings.pnad.employment
    builder = (
        SidraQueryBuilder(cfg.table_id)
        .select_periods(cfg.period)
        .select_variables(cfg.variable_id)
        .select_locations(cfg.location_level)
    )
    for c in cfg.classifications:
        builder.select_classification(c.id, c.categories)

    data = sidra_client.get_data(builder)

    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert isinstance(item, SidraVariableResponse)
        assert item.resultados


def test_pns_health_assessment_query(sidra_client):
    """
    Test real PNS query (Table 7666) for health assessment.
    """
    cfg = settings.pns.health_assessment
    builder = (
        SidraQueryBuilder(cfg.table_id)
        .select_periods(cfg.period)
        .select_variables(cfg.variable_id)
        .select_locations(cfg.location_level)
    )
    for c in cfg.classifications:
        builder.select_classification(c.id, c.categories)

    data = sidra_client.get_data(builder)

    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert isinstance(item, SidraVariableResponse)
        assert item.resultados
