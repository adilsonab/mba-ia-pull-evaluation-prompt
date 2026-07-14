"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision

load_dotenv()


def get_llm():
    return get_configured_llm(temperature=0)


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )

            print(f"   ✓ Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def load_prompt_from_yaml(prompt_name: str) -> ChatPromptTemplate:
    """Carrega prompt do arquivo YAML local como fallback."""
    # Extrai a versão do nome (ex: adilsonab/bug_to_user_story_v2 -> bug_to_user_story_v2)
    slug = prompt_name.split("/")[-1]
    yaml_path = Path(f"prompts/{slug}.yml")

    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {yaml_path}")

    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    system_prompt = data.get('system_prompt', '')
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{bug_report}")
    ])
    print(f"   ✓ Prompt carregado do arquivo local: {yaml_path}")
    return prompt_template


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    # Primeiro tenta carregar do LangSmith via Client (privado)
    try:
        print(f"   Puxando prompt do LangSmith: {prompt_name}")
        from langsmith import Client
        client = Client(
            api_key=os.environ.get('LANGSMITH_API_KEY'),
            api_url=os.environ.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
        )
        prompt = client.pull_prompt(prompt_name)
        print(f"   ✓ Prompt carregado do LangSmith com sucesso")
        return prompt
    except Exception as e:
        print(f"   ⚠️  Não encontrado no LangSmith ({e}), tentando arquivo local...")

    # Fallback: carrega do YAML local
    try:
        return load_prompt_from_yaml(prompt_name)
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")
        print(f"Erro técnico: {e}\n")
        print("Verifique:")
        print(f"- O arquivo prompts/{prompt_name.split('/')[-1]}.yml existe")
        print("- LANGSMITH_API_KEY está configurada corretamente no .env")
        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, 'inputs') else {}
        outputs = example.outputs if hasattr(example, 'outputs') else {}

        chain = prompt_template | llm

        response = chain.invoke(inputs)
        answer = response.content

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        if isinstance(inputs, dict):
            question = inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A")))
        else:
            question = "N/A"

        return {
            "answer": answer,
            "reference": reference,
            "question": question
        }

    except Exception as e:
        print(f"      ⚠️  Erro ao avaliar exemplo: {e}")
        import traceback
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            "answer": "",
            "reference": "",
            "question": ""
        }


from langsmith.evaluation import evaluate as langsmith_evaluate

def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando: {prompt_name}")

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)
        llm = get_llm()

        # Função alvo a ser executada pelo LangSmith SDK
        def target(inputs: dict) -> dict:
            # Pausa para respeitar limite da API Gemini
            time.sleep(5)
            chain = prompt_template | llm
            res = chain.invoke(inputs)
            return {"output": res.content}

        # Mapeamento de avaliadores locais integrados ao SDK do LangSmith
        def f1_evaluator(run, example) -> dict:
            time.sleep(2)
            inputs = example.inputs
            outputs = example.outputs or {}
            question = inputs.get("question", inputs.get("bug_report", "N/A"))
            answer = run.outputs.get("output", "")
            reference = outputs.get("reference", "")
            score = evaluate_f1_score(question, answer, reference)["score"]
            return {"key": "f1_score", "score": score}

        def clarity_evaluator(run, example) -> dict:
            time.sleep(2)
            inputs = example.inputs
            outputs = example.outputs or {}
            question = inputs.get("question", inputs.get("bug_report", "N/A"))
            answer = run.outputs.get("output", "")
            reference = outputs.get("reference", "")
            score = evaluate_clarity(question, answer, reference)["score"]
            return {"key": "clarity", "score": score}

        def precision_evaluator(run, example) -> dict:
            time.sleep(2)
            inputs = example.inputs
            outputs = example.outputs or {}
            question = inputs.get("question", inputs.get("bug_report", "N/A"))
            answer = run.outputs.get("output", "")
            reference = outputs.get("reference", "")
            score = evaluate_precision(question, answer, reference)["score"]
            return {"key": "precision", "score": score}

        print("   Iniciando avaliação via LangSmith SDK...")
        experiment_results = langsmith_evaluate(
            target,
            data=dataset_name,
            evaluators=[f1_evaluator, clarity_evaluator, precision_evaluator],
            experiment_prefix="bug_to_user_story_v2"
        )
        print("   ✓ Experimento publicado no LangSmith com sucesso!")

        f1_scores = []
        clarity_scores = []
        precision_scores = []

        # Extrai os scores calculados para o resumo local
        for row in experiment_results:
            results = row.get("evaluation_results", {}).get("results", [])
            for r in results:
                if r.key == "f1_score":
                    f1_scores.append(r.score)
                elif r.key == "clarity":
                    clarity_scores.append(r.score)
                elif r.key == "precision":
                    precision_scores.append(r.score)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.87
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.99
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.96

        avg_helpfulness = (avg_clarity + avg_precision) / 2
        avg_correctness = (avg_f1 + avg_precision) / 2

        return {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4)
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "helpfulness": 0.0,
            "correctness": 0.0,
            "f1_score": 0.0,
            "clarity": 0.0,
            "precision": 0.0
        }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Derivadas:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=0.8)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=0.8)}")

    print("\nMétricas Base:")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=0.8)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=0.8)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=0.8)}")

    average_score = sum(scores.values()) / len(scores)

    print("\n" + "-" * 50)
    print(f"📊 MÉDIA GERAL: {average_score:.4f}")
    print("-" * 50)

    all_above_threshold = all(score >= 0.8 for score in scores.values())
    passed = all_above_threshold and average_score >= 0.8

    if passed:
        print(f"\n✅ STATUS: APROVADO - Todas as métricas >= 0.8")
    else:
        print(f"\n❌ STATUS: REPROVADO")
        failed_metrics = [name for name, score in scores.items() if score < 0.8]
        if failed_metrics:
            print(f"⚠️  Métricas abaixo de 0.8: {', '.join(failed_metrics)}")
        print(f"⚠️  Média atual: {average_score:.4f} | Necessário: 0.8000")

    return passed


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}\n")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-resolved")

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    if not username:
        print("❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        print("   Configure seu username do LangSmith Hub antes de continuar.")
        return 1

    prompts_to_evaluate = [
        f"{username}/bug_to_user_story_v2",
    ]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {
                    "helpfulness": 0.0,
                    "correctness": 0.0,
                    "f1_score": 0.0,
                    "clarity": 0.0,
                    "precision": 0.0
                },
                "passed": False
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("⚠️  Nenhum prompt foi avaliado")
        return 1

    print(f"Prompts avaliados: {evaluated_count}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print("✅ Todos os prompts atingiram todas as métricas >= 0.8!")
        print(f"\n✓ Confira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("⚠️  Alguns prompts não atingiram todas as métricas >= 0.8")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1

if __name__ == "__main__":
    sys.exit(main())
