import wikipediaapi
import random
import json

wiki = wikipediaapi.Wikipedia(
    language='pt',
    extract_format=wikipediaapi.ExtractFormat.WIKI,  # Formato do texto
    user_agent="Mozilla/5.0 (pvasx123@gmail.com)"  # Identificação

)

# Página da categoria "Matemática"
categoria = wiki.page("Categoria:Matemática")

def extrair_artigos_da_categoria(categoria, depth=0):
    artigos = []
    # Pega todos os artigos da categoria
    i = 0
    for page in categoria.categorymembers.values():
        print(i, depth)
        i += 1
        if page.ns == wikipediaapi.Namespace.MAIN:  # Filtra apenas artigos (não subcategorias)
            artigos.append(page)
        elif page.ns == wikipediaapi.Namespace.CATEGORY and depth <= 2:  # Entra em subcategorias recursivamente
            artigos += extrair_artigos_da_categoria(page, depth+1)
    return artigos

def salvar_dados(artigos):
    data = {
        'train' : [],
        'eval' :  []
    }

    train_idx = random.sample(list(range(len(artigos))), int(len(artigos)*0.7))
    test_idx = [i for i in range(len(artigos)) if i not in train_idx]

    data['train'] = train_idx
    data['eval'] = test_idx

    with open(os.path.join("data", "train.json"), 'w') as json_file:
        json.dump(json_file, data)
