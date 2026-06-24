from infrastructure.llm import BaseLLMClient
from infrastructure.llm.models import ModelAnswer
from features.generate_persona.models import (
    Persona,
    DemographicProfile,
    EconomicProfile,
    SocialProfile,
    HealthProfile,
)
from features.opinion_simulation.models import SimulationConfig, Survey, SurveyQuestion
from features.opinion_simulation.handler import run_simulation


class MockClient(BaseLLMClient):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.question_answer_calls = []

    def generate(self, prompt, system_prompt=None, json_schema=None):
        return "mock"

    def question_answer(self, system, question, options, messages=None, **kwargs):
        self.question_answer_calls.append(
            {
                "system": system,
                "question": question,
                "messages": messages.copy() if messages else [],
            }
        )
        return ModelAnswer(answer=("a", "Concordo"), explanation="Mocked")


def test_run_simulation(monkeypatch):
    # Setup mock factory
    mock_client = MockClient("mock-model")
    monkeypatch.setattr(
        "features.opinion_simulation.handler.LLMFactory.provide",
        lambda self, model_id: mock_client,
    )

    # Create mock Persona
    import uuid

    persona_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    persona = Persona(
        id=persona_id,
        demographic=DemographicProfile(
            age_group="18-24", gender="F", race="Branca", region="Sul", urban=True
        ),
        economic=EconomicProfile(
            education_level="Superior",
            employment_status="Empregado",
            income_per_capita="2SM",
            inflation_rate=0.5,
        ),
        social=SocialProfile(marital_status="Solteiro", religion="Católica"),
        health=HealthProfile(health_self_assessment="Boa", has_chronic_disease=False),
    )

    # Create configuration
    config = SimulationConfig(
        personas=1,
        repetitions=2,
        models=["mock-model"],
        results_path="/tmp",
        surveys=[
            Survey(
                id="s1",
                title="Test Survey",
                questions=[
                    SurveyQuestion(
                        id="q1",
                        topic="T",
                        text="Q1",
                        options={"a": "Concordo", "b": "Discordo"},
                    ),
                    SurveyQuestion(
                        id="q2", topic="T", text="Q2", options={"a": "Sim", "b": "Nao"}
                    ),
                ],
            )
        ],
    )

    # Run
    results = run_simulation(config=config, personas=[persona])

    # Assertions
    assert len(results) == 1
    assert results[0].persona.id == persona_id
    assert len(results[0].responses) == 2  # 2 questions

    resp_q1 = results[0].responses[0]
    assert resp_q1.question == "Q1"
    assert len(resp_q1.answers) == 2  # 2 repetitions
    assert resp_q1.result.distribution["a"] == 1.0  # Mock always answers "a"
    assert resp_q1.result.distribution["b"] == 0.0

    # Check messages history accumulation (2 repetitions, 2 questions each = 4 calls total)
    assert len(mock_client.question_answer_calls) == 4

    # Check first repetition, second question
    call_q2 = mock_client.question_answer_calls[1]
    assert call_q2["question"] == "Q2"
    assert len(call_q2["messages"]) == 2  # Should contain assistant/user from Q1

    # Check second repetition, first question
    call_rep2_q1 = mock_client.question_answer_calls[2]
    assert call_rep2_q1["question"] == "Q1"
    assert (
        len(call_rep2_q1["messages"]) == 0
    )  # History should be reset for new repetition
