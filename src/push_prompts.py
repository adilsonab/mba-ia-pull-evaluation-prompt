"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')
os.environ['LANGCHAIN_HUB_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    try:
        print(f"Fazendo push do prompt: {prompt_name}...")
        from langsmith import Client

        client = Client(
            api_key=os.environ.get('LANGSMITH_API_KEY'),
            api_url=os.environ.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
        )

        system_prompt = prompt_data.get('system_prompt', '')
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{bug_report}")
        ])

        url = client.push_prompt(
            prompt_name,
            object=prompt_template,
            is_public=False,  # privado para não precisar de Hub Handle
            description=prompt_data.get('description', ''),
            tags=prompt_data.get('techniques_applied', [])
        )
        print(f"✅ Prompt salvo com sucesso: {url}")
        print(f"   (Salvo como privado no seu workspace LangSmith)")
        return True
    except Exception as e:
        print(f"❌ Erro ao fazer push do prompt: {e}")
        return False

def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    errors = []

    required_fields = ['description', 'system_prompt', 'version']
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Campo obrigatório faltando: {field}")

    system_prompt = prompt_data.get('system_prompt', '').strip()
    if not system_prompt:
        errors.append("system_prompt está vazio")

    if 'TODO' in system_prompt:
        errors.append("system_prompt ainda contém TODOs")

    techniques = prompt_data.get('techniques_applied', [])
    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    return (len(errors) == 0, errors)

def main():
    print_section_header("Push de Prompts para o LangSmith")
    
    if not check_env_vars(['LANGSMITH_API_KEY']):
        return 1
        
    prompt_file = "prompts/bug_to_user_story_v2.yml"
    prompt_data = load_yaml(prompt_file)
    
    if not prompt_data:
        return 1
        
    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Validação falhou:")
        for err in errors:
            print(f"  - {err}")
        return 1
        
    # Tenta com username/handle primeiro, depois tenta só o nome base
    username = os.getenv('USERNAME_LANGSMITH_HUB', '')
    prompt_name_with_user = f"{username}/bug_to_user_story_v2" if username else "bug_to_user_story_v2"
    prompt_name_base = "bug_to_user_story_v2"

    # Tenta com prefixo de usuário
    success = push_prompt_to_langsmith(prompt_name_with_user, prompt_data)

    # Se falhar (sem Hub Handle), tenta só o nome base
    if not success:
        print("\n⚠️  Tentando salvar sem prefixo de usuário...")
        success = push_prompt_to_langsmith(prompt_name_base, prompt_data)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
