"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')
os.environ['LANGCHAIN_HUB_API_KEY'] = os.getenv('LANGSMITH_API_KEY', '')

from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header


def pull_prompts_from_langsmith():
    print_section_header("Pull de Prompts do LangSmith")
    
    if not check_env_vars(['LANGSMITH_API_KEY']):
        return False
        
    prompt_name = "leonanluppi/bug_to_user_story_v1"
    output_path = "prompts/bug_to_user_story_v1.yml"
    
    try:
        print(f"Fazendo pull do prompt: {prompt_name}...")
        prompt = hub.pull(
            prompt_name,
            api_key=os.environ.get('LANGSMITH_API_KEY')
        )
        
        # Save prompt to file
        prompt_data = {
            "name": prompt_name,
            "description": "Transforma um relato de bug em uma User Story.",
            "version": "1.0",
            "system_prompt": prompt.messages[0].prompt.template if len(prompt.messages)>0 else "",
            "techniques_applied": []
        }
        
        if save_yaml(prompt_data, output_path):
            print(f"✅ Prompt salvo com sucesso em: {output_path}")
            return True
            
    except Exception as e:
        print(f"❌ Erro ao fazer pull do prompt: {e}")
        return False

def main():
    """Função principal"""
    pull_prompts_from_langsmith()

if __name__ == "__main__":
    sys.exit(main())
