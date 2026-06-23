from features.generate_persona import generate_one, generate_batch
from features.generate_persona.models import Persona

def test_generate_one_integration(sidra_client, db_cache):
    """
    Test generate_one against distribution cache and SIDRA query build pipeline.
    """
    persona = generate_one(sidra_client, db_cache)
    
    assert isinstance(persona, Persona)
    assert persona.id is not None
    assert persona.demographic.age_group != ""
    assert persona.economic.education_level != ""
    assert persona.health.health_self_assessment != ""
    assert persona.social.religion != ""

def test_generate_batch_integration(sidra_client, db_cache):
    """
    Test generate_batch works successfully.
    """
    batch = generate_batch(sidra_client, db_cache, count=3)
    
    assert len(batch) == 3
    for persona in batch:
        assert isinstance(persona, Persona)
        assert persona.id is not None
