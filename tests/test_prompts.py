"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TestPrompts:
    def test_prompt_has_system_prompt(self):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        prompt_data = load_prompts("prompts/bug_to_user_story_v2.yml")
        assert "system_prompt" in prompt_data
        assert bool(prompt_data["system_prompt"].strip())

    def test_prompt_has_role_definition(self):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        prompt_data = load_prompts("prompts/bug_to_user_story_v2.yml")
        system_prompt = prompt_data.get("system_prompt", "").lower()
        assert "você é" in system_prompt or "você atua como" in system_prompt or "como um" in system_prompt

    def test_prompt_mentions_format(self):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        prompt_data = load_prompts("prompts/bug_to_user_story_v2.yml")
        system_prompt = prompt_data.get("system_prompt", "").lower()
        assert "markdown" in system_prompt or "user story" in system_prompt

    def test_prompt_has_few_shot_examples(self):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        prompt_data = load_prompts("prompts/bug_to_user_story_v2.yml")
        system_prompt = prompt_data.get("system_prompt", "").lower()
        assert "exemplo" in system_prompt or "examples" in system_prompt or "few-shot" in system_prompt

    def test_prompt_no_todos(self):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        prompt_data = load_prompts("prompts/bug_to_user_story_v2.yml")
        system_prompt = prompt_data.get("system_prompt", "")
        assert "[TODO]" not in system_prompt

    def test_minimum_techniques(self):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        prompt_data = load_prompts("prompts/bug_to_user_story_v2.yml")
        techniques = prompt_data.get("techniques_applied", [])
        assert isinstance(techniques, list)
        assert len(techniques) >= 2

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])