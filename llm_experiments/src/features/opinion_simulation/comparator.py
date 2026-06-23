import json
from typing import List, Dict
from .models import SimulationComparison


def export_comparison_to_json(
    comparisons: List[SimulationComparison], output_path: str
):
    """
    Export the detailed comparison data to a JSON file.
    """
    data = [comp.model_dump(mode="json") for comp in comparisons]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_summary_report(comparisons: List[SimulationComparison]) -> str:
    """
    Generates a markdown summary comparing how different models answered the questions.
    """
    lines = ["# Relatório de Simulação Comparativa de Opinião\n"]

    for comp in comparisons:
        q = comp.question
        lines.append(f"## Tópico: {q.topic}")
        lines.append(f"**Pergunta:** {q.question_text}\n")

        # Aggregate answers by model
        # model_id -> option -> count
        stats: Dict[str, Dict[str, int]] = {}
        total_personas = len(comp.persona_results)

        for res in comp.persona_results:
            for m_res in res.responses:
                model = m_res.model_id
                option = m_res.chosen_option
                if model not in stats:
                    stats[model] = {}
                stats[model][option] = stats[model].get(option, 0) + 1

        lines.append(f"### Estatísticas (Base: {total_personas} personas)")
        for model_id, counts in stats.items():
            lines.append(f"\n**Modelo: {model_id}**")
            for opt in q.options:
                c = counts.get(opt, 0)
                pct = (c / total_personas) * 100 if total_personas > 0 else 0
                lines.append(f"- {opt}: {c} ({pct:.1f}%)")

        lines.append("\n---\n")

    return "\n".join(lines)
