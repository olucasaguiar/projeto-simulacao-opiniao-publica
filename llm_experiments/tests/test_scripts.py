import json
import sys
import pytest
import yaml
from pathlib import Path

# Add src to system path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import create_personas
import create_batch


def test_create_personas_cli(tmp_path, monkeypatch):
    """
    Test the create_personas.py CLI script by patching sys.argv and running main().
    """
    output_file = tmp_path / "personas.jsonl"
    
    # Mock sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        ["create_personas.py", "2", str(output_file)],
    )

    # Run CLI
    create_personas.main()

    # Assertions
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    assert len(lines) == 2
    
    # Check persona structure
    for line in lines:
        persona = json.loads(line)
        assert "id" in persona
        assert "demographic" in persona
        assert "economic" in persona
        assert "health" in persona
        assert "social" in persona
        assert persona["demographic"]["gender"] in ["Masculino", "Feminino"]


def test_create_batch_cli(tmp_path, monkeypatch):
    """
    Test the create_batch.py CLI script by patching sys.argv and running main().
    """
    # 1. Create a dummy personas file
    personas_file = tmp_path / "personas.jsonl"
    dummy_personas = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "demographic": {"age_group": "18-24", "gender": "Feminino", "race": "Parda", "region": "Norte", "urban": False},
            "economic": {"education_level": "Superior", "employment_status": "Ocupada", "income_per_capita": "1SM", "inflation_rate": 0.5},
            "health": {"health_self_assessment": "Bom", "has_chronic_disease": False},
            "social": {"marital_status": "Solteiro(a)", "religion": "Católica"}
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "demographic": {"age_group": "25-34", "gender": "Masculino", "race": "Branca", "region": "Sul", "urban": True},
            "economic": {"education_level": "Médio", "employment_status": "Desocupada", "income_per_capita": "2SM", "inflation_rate": 0.5},
            "health": {"health_self_assessment": "Regular", "has_chronic_disease": True},
            "social": {"marital_status": "Casado(a)", "religion": "Evangélica"}
        }
    ]
    with open(personas_file, "w", encoding="utf-8") as f:
        for p in dummy_personas:
            f.write(json.dumps(p) + "\n")

    # 2. Create YAML template file
    template_file = tmp_path / "template.yaml"
    template_content = {
        "model": "gpt-4o-mini",
        "temperature": 0.8,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
        "system": "Você é do gênero {{ gender }} e mora na região {{ region }}.",
        "messages": [
            {
                "role": "user",
                "content": "Olá! Eu sou {{ marital_status }} e moro na zona {{ urban_str }}."
            }
        ]
    }
    with open(template_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(template_content, f)

    # 3. Mock sys.argv for create_batch
    output_file = tmp_path / "batch.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_batch.py",
            str(template_file),
            str(personas_file),
            str(output_file)
        ]
    )

    # Run CLI
    create_batch.main()

    # Assertions
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2

    # Verify line 1 (Feminino, Norte, rural, Solteiro(a))
    record_1 = json.loads(lines[0])
    assert record_1["custom_id"] == "11111111-1111-1111-1111-111111111111"
    assert record_1["method"] == "POST"
    assert record_1["url"] == "/v1/chat/completions"
    
    body_1 = record_1["body"]
    assert body_1["model"] == "gpt-4o-mini"
    assert body_1["temperature"] == 0.8  # Resolved from template
    assert body_1["max_tokens"] == 120    # Template default
    assert body_1["response_format"] == {"type": "json_object"}
    
    messages_1 = body_1["messages"]
    assert len(messages_1) == 2
    assert messages_1[0]["role"] == "system"
    assert messages_1[0]["content"] == "Você é do gênero Feminino e mora na região Norte."
    assert messages_1[1]["role"] == "user"
    assert messages_1[1]["content"] == "Olá! Eu sou Solteiro(a) e moro na zona Rural."

    # Verify line 2 (Masculino, Sul, urban, Casado(a))
    record_2 = json.loads(lines[1])
    assert record_2["custom_id"] == "22222222-2222-2222-2222-222222222222"
    
    body_2 = record_2["body"]
    messages_2 = body_2["messages"]
    assert len(messages_2) == 2
    assert messages_2[0]["role"] == "system"
    assert messages_2[0]["content"] == "Você é do gênero Masculino e mora na região Sul."
    assert messages_2[1]["role"] == "user"
    assert messages_2[1]["content"] == "Olá! Eu sou Casado(a) e moro na zona Urbana."


def test_create_batch_continuation(tmp_path, monkeypatch):
    """
    Test the create_batch.py CLI script with multi-turn messages and previous turn response parsing.
    """
    # 1. Create a dummy personas file
    personas_file = tmp_path / "personas_cont.jsonl"
    dummy_personas = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "demographic": {"age_group": "18-24", "gender": "Feminino", "race": "Parda", "region": "Norte", "urban": False},
            "economic": {"education_level": "Superior", "employment_status": "Ocupada", "income_per_capita": "1SM", "inflation_rate": 0.5},
            "health": {"health_self_assessment": "Bom", "has_chronic_disease": False},
            "social": {"marital_status": "Solteiro(a)", "religion": "Católica"}
        }
    ]
    with open(personas_file, "w", encoding="utf-8") as f:
        for p in dummy_personas:
            f.write(json.dumps(p) + "\n")

    # 2. Create previous batch results file containing 2 replications for this persona
    prev_responses_file = tmp_path / "prev_responses.jsonl"
    prev_data = [
        {
            "custom_id": "11111111-1111-1111-1111-111111111111_P02_rep1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "{\"answer\": \"a\", \"explanation\": \"Sim\"}"
                            }
                        }
                    ]
                }
            }
        },
        {
            "custom_id": "11111111-1111-1111-1111-111111111111_P02_rep2",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "{\"answer\": \"b\", \"explanation\": \"Não\"}"
                            }
                        }
                    ]
                }
            }
        }
    ]
    with open(prev_responses_file, "w", encoding="utf-8") as f:
        for r in prev_data:
            f.write(json.dumps(r) + "\n")

    # 3. Create YAML template file referencing the prev_responses path
    template_file = tmp_path / "template_cont.yaml"
    template_content = {
        "model": "gpt-4o-mini",
        "system": "Você é do gênero {{ gender }}.",
        "messages": [
            {
                "role": "user",
                "content": "Primeira pergunta?"
            },
            {
                "role": "assistant",
                "path": str(prev_responses_file)
            },
            {
                "role": "user",
                "content": "Segunda pergunta para quem mora no {{ region }}?"
            }
        ]
    }
    with open(template_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(template_content, f)

    # 4. Mock sys.argv
    output_file = tmp_path / "batch_cont.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_batch.py",
            str(template_file),
            str(personas_file),
            str(output_file)
        ]
    )

    # Run CLI
    create_batch.main()

    # Assertions
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Should generate 2 separate request entries, one for each replication branch
    assert len(lines) == 2

    # Verify line 1
    record_1 = json.loads(lines[0])
    assert record_1["custom_id"] == "11111111-1111-1111-1111-111111111111_P02_rep1"
    messages_1 = record_1["body"]["messages"]
    assert len(messages_1) == 4
    assert messages_1[0] == {"role": "system", "content": "Você é do gênero Feminino."}
    assert messages_1[1] == {"role": "user", "content": "Primeira pergunta?"}
    assert messages_1[2] == {"role": "assistant", "content": "{\"answer\": \"a\", \"explanation\": \"Sim\"}"}
    assert messages_1[3] == {"role": "user", "content": "Segunda pergunta para quem mora no Norte?"}

    # Verify line 2
    record_2 = json.loads(lines[1])
    assert record_2["custom_id"] == "11111111-1111-1111-1111-111111111111_P02_rep2"
    messages_2 = record_2["body"]["messages"]
    assert len(messages_2) == 4
    assert messages_2[0] == {"role": "system", "content": "Você é do gênero Feminino."}
    assert messages_2[1] == {"role": "user", "content": "Primeira pergunta?"}
    assert messages_2[2] == {"role": "assistant", "content": "{\"answer\": \"b\", \"explanation\": \"Não\"}"}
    assert messages_2[3] == {"role": "user", "content": "Segunda pergunta para quem mora no Norte?"}
