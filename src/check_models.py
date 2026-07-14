import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Aviso: GOOGLE_API_KEY não está configurada no .env!")
    exit(1)

genai.configure(api_key=api_key)

try:
    print("Consultando modelos de CHAT disponíveis...")
    models = genai.list_models()
    found = False
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"Modelo de chat suportado: {m.name}")
            found = True
            
    if not found:
        print("Nenhum modelo de chat suportado foi encontrado.")
except Exception as e:
    print(f"Erro ao consultar API do Google: {e}")
