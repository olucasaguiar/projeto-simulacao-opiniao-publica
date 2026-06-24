from features.generate_persona.models import Persona
from .models import SurveyQuestion


def build_persona_system_prompt(persona: Persona) -> str:
    """
    Builds a robust, natural language system prompt from the Persona object.
    It contextualizes the AI into the persona's socio-economic and demographic profile.
    """
    demo = persona.demographic
    eco = persona.economic
    health = persona.health
    social = persona.social

    urban_str = "Urbana" if demo.urban else "Rural"

    prompt = (
        f"Você é um cidadão brasileiro com o seguinte perfil:\n"
        f"- Idade: {demo.age_group}\n"
        f"- Gênero: {demo.gender}\n"
        f"- Raça/Cor: {demo.race}\n"
        f"- Região do Brasil: {demo.region} (Zona {urban_str})\n"
        f"- Escolaridade: {eco.education_level}\n"
        f"- Situação de emprego: {eco.employment_status}\n"
        f"- Faixa de renda: {eco.income_per_capita}\n"
        f"- Estado civil: {social.marital_status}\n"
        f"- Religião: {social.religion}\n"
        f"- Autoavaliação de saúde: {health.health_self_assessment}\n\n"
        "Sua tarefa é responder a perguntas de opinião pública como se você fosse essa pessoa. "
        "Considere as condições socioeconômicas e demográficas descritas para formar sua opinião, refletindo "
        "o contexto real do Brasil. Não saia do personagem. Responda de forma sincera baseando-se nas experiências "
        "prováveis de alguém com este exato perfil."
    )
    return prompt


def build_question_prompt(question: SurveyQuestion) -> str:
    """
    Builds the user prompt for the given question.
    """
    options_text = "\n".join(
        [f"{key}) {value}" for key, value in question.options.items()]
    )
    prompt = (
        f"Tópico: {question.topic}\n"
        f"Pergunta: {question.text}\n\n"
        f"Alternativas:\n{options_text}\n\n"
        "Responda APENAS com um JSON válido no formato:\n"
        '{"answer_key": "letra_escolhida", "answer_value": "texto_da_alternativa", "explanation": "sua justificativa"}\n'
        "Não adicione nenhum texto antes ou depois do JSON."
    )
    return prompt
