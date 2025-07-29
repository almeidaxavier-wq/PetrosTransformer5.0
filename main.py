from transformers import AutoModel, AutoTokenizer, pipeline
from donwload_wikipedia import extrair_artigos_da_categoria, salvar_dados
from trainer import treinar_com_artigos
import markdown, wikipediaapi
import os, json, pickle

wiki = wikipediaapi.Wikipedia(
    language='pt',
    extract_format=wikipediaapi.ExtractFormat.WIKI,  # Formato do texto
    user_agent="Mozilla/5.0 (pvasx123@gmail.com)"  # Identificação

)

# Página da categoria "Matemática"
categoria = wiki.page("Categoria:Matemática")

def init():
    artigos = []
    if os.path.exists(os.path.join("data", "artigos.pkl")):
        with open("data/artigos.pkl", 'rb') as file:
            artigos = pickle.load(file)

    elif not os.path.exists(os.path.join("data", "train.json")):
        artigos = extrair_artigos_da_categoria(categoria)
        salvar_dados(artigos)

    if not os.path.exists(os.path.join('data', 'artigos.pkl')):
        with open("data/artigos.pkl", 'wb') as file:
            pickle.dump(artigos, file)

    dados = {}

    with open(os.path.join("data", "train.json"), 'r') as json_file:
        dados = json.load(json_file)

    treinar_com_artigos(dados, artigos)
    modelo = AutoModel.from_pretrained(os.path.join("models", 'fine-tune-petros'))
    tokenizer = AutoTokenizer().from_pretrained(os.path.join("models", 'fine-tune-petros'))

    return modelo, tokenizer

def mostrar_resposta(resposta, arquivo_html):
    with open(arquivo_html, 'w') as file:
        file.write(markdown.markdown(resposta))

def calcular_resposta(query, modelo, tokenizer):
    final_query= f"""
    Assistente: Você é um matemático altamente conceituado e precisa tirar as dúvidas de seus alunos, ajudando-os a provar teoremas ou até conjecturas.
    Trate as conjecturas apenas como teoremas a serem provados ou refutados e use toda a sua fonte de informação para responder às peguntas. Caso não consiga encontrar uma resposta plausível, mostre os pontos importantes 
    da sua tentativa de demonstração, não tem problema. No mais, seja consiso e formal em sua resposta, repondendo tudo em markdown. 
    
    Pergunta: {query} 
    """
    translator_pt_en = pipeline("translation", model="Helsinki-NLP/opus-mt-pt-en")
    translator_en_pt = pipeline("translation", model="Helsinki-NLP/opus-mt-en-pt")
    translated_final_query = translator_pt_en(final_query)[0]['translation_text']

    encoded_inputs = tokenizer(translated_final_query, return_tensors="en", padding=True, truncation=True)
    outputs = modelo(encoded_inputs, max_length=200, num_return_sequences=1)

    mostrar_resposta(translator_en_pt(outputs[0]['generated_text'])[0]['translation_text'], 'output.html')

if __name__ == '__main__':
    query = "Eu posso dizer que, para todo o número par, existem dois primos que o compõem?"

    model, tokenizer = init()
    calcular_resposta(query, model, tokenizer)
