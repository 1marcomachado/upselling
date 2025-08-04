import xml.etree.ElementTree as ET
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json
from datetime import datetime
import base64
from dotenv import load_dotenv
from collections import defaultdict

# 🔐 Carregar variáveis de ambiente
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
repo = os.getenv("GITHUB_REPO")
branch = "main"
filename = "upselling_final.json"
api_url = f"https://api.github.com/repos/{repo}/contents/{filename}"
xml_url = "https://www.bzronline.com/extend/catalog_24.xml"
image_folder = "imagens"
os.makedirs(image_folder, exist_ok=True)

# 📁 Categorias complementares
with open("categorias_complementares.json", "r", encoding="utf-8") as f:
    categorias_complementares = json.load(f)

# 🌐 Sessão com retry
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# 🔽 Descarregar XML
print("🔽 A descarregar XML...")
r = session.get(xml_url, timeout=30)
r.raise_for_status()
xml_content = r.content
tree = ET.ElementTree(ET.fromstring(xml_content))
root = tree.getroot()
ns = {'g': 'http://base.google.com/ns/1.0', 'atom': 'http://www.w3.org/2005/Atom'}

# 📦 Parse produtos
produtos = []
for entry in root.findall('atom:entry', ns):
    id_ = entry.find('g:id', ns).text if entry.find('g:id', ns) is not None else ""
    mpn = entry.find('g:mpn', ns).text if entry.find('g:mpn', ns) is not None else ""
    title = entry.find('g:title', ns).text.strip() if entry.find('g:title', ns) is not None else ""
    category = entry.find('g:category', ns).text if entry.find('g:category', ns) is not None else ""
    brand = entry.find('g:brand', ns).text.strip() if entry.find('g:brand', ns) is not None else ""
    image_link = entry.find('g:image_link', ns).text if entry.find('g:image_link', ns) is not None else ""
    site = entry.find('g:site', ns).text if entry.find('g:site', ns) is not None else ""
    gender = entry.find('g:gender', ns).text if entry.find('g:gender', ns) is not None else ""
    pack = entry.find('g:pack', ns).text if entry.find('g:pack', ns) is not None else ""
    price = entry.find('g:price', ns).text if entry.find('g:price', ns) is not None else ""
    sale_price = entry.find('g:sale_price', ns).text if entry.find('g:sale_price', ns) is not None else ""
    size = entry.find('g:size', ns).text if entry.find('g:size', ns) is not None else ""
    availability = entry.find('g:availability', ns).text if entry.find('g:availability', ns) is not None else ""

    produtos.append({
        "id": id_,
        "mpn": mpn,
        "title": title,
        "price": price,
        "sale_price": sale_price,
        "category": category,
        "brand": brand,
        "image_link": image_link,
        "site": site,
        "gender": gender,
        "pack": pack,
        "size": size,
        "availability": availability
    })

# 🧠 Modelo ResNet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval().to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📸 Função de embedding de imagem
def get_image_embedding(img_url, image_path):
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
        else:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(img_url, headers=headers, timeout=10)
            if "image" not in response.headers.get("Content-Type", ""):
                print(f"❌ Ignorado (não é imagem): {img_url}")
                return None
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(image_path)
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"⚠️ Erro ao processar imagem {img_url}: {e}")
        return None

# 🧠 Preparar embeddings com cache (versão corrigida)
print("🧠 A preparar embeddings com cache...")
embeddings_file = "embeddings.npy"
mpns_file = "mpns_embeddings.json"

if os.path.exists(embeddings_file) and os.path.exists(mpns_file):
    embeddings = np.load(embeddings_file)
    with open(mpns_file, "r", encoding="utf-8") as f:
        mpns_existentes = json.load(f)
else:
    embeddings = []
    mpns_existentes = []

mpn_to_embedding = {}
mpn_to_produto = {}

# Produtos com embeddings existentes
for i, mpn in enumerate(mpns_existentes):
    mpn_to_embedding[mpn] = embeddings[i]

# Novos produtos
novos_embeddings = []
mpns_adicionados = set(mpns_existentes)

for p in produtos:
    mpn = p["mpn"]
    if not mpn or mpn in mpns_adicionados:
        continue

    image_filename = f"{mpn}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    emb = get_image_embedding(p["image_link"], image_path)

    if emb is not None:
        novos_embeddings.append(emb)
        mpn_to_embedding[mpn] = emb
        mpn_to_produto[mpn] = p
        mpns_adicionados.add(mpn)

# Atualizar embeddings
if novos_embeddings:
    novos_embeddings = np.array(novos_embeddings)
    embeddings = np.vstack((embeddings, novos_embeddings)) if len(embeddings) else novos_embeddings
    np.save(embeddings_file, embeddings)
    with open(mpns_file, "w", encoding="utf-8") as f:
        json.dump(list(mpns_adicionados), f, ensure_ascii=False, indent=2)

# Reconstruir produtos válidos com embeddings
produtos_validos = []
for p in produtos:
    mpn = p["mpn"]
    if mpn in mpn_to_embedding:
        produtos_validos.append(p)
        mpn_to_produto[mpn] = p

print(f"✅ Produtos com embeddings: {len(produtos_validos)}")

# 🔎 Similaridade por mpn
print("📊 A calcular similaridades...")
mpn_list = list(mpn_to_embedding.keys())
mpn_embeddings = np.array([mpn_to_embedding[m] for m in mpn_list])
similarity_matrix = cosine_similarity(mpn_embeddings)

# 🔄 Construir sugestões
sugestoes_dict = {}
produtos_sem_sugestoes = []
for p in produtos_validos:
    base_mpn = p["mpn"]
    if base_mpn not in mpn_to_embedding:
        continue
    i = mpn_list.index(base_mpn)
    last_level_base = p["category"].split(">")[-1].strip()
    second_level_base = p["category"].split(">")[1].strip() if len(p["category"].split(">")) > 1 else ""

    candidatos = [
        j for j, mpn_cand in enumerate(mpn_list)
        if mpn_cand != base_mpn and
           mpn_to_produto[mpn_cand]["site"] == p["site"] and
           mpn_to_produto[mpn_cand]["brand"] == p["brand"] and
           mpn_to_produto[mpn_cand]["gender"] == p["gender"]
    ]

    if last_level_base in categorias_complementares:
        categorias_validas = categorias_complementares[last_level_base]
        candidatos = [j for j in candidatos if mpn_to_produto[mpn_list[j]]["category"].split(">")[-1].strip() in categorias_validas]

    if candidatos:
        scores = similarity_matrix[i][candidatos]
        indices_ordenados = np.argsort(scores)[::-1][:16]
        sugestoes = [mpn_list[candidatos[k]] for k in indices_ordenados]
        sugestoes_dict[p["id"]] = sugestoes
    else:
        produtos_sem_sugestoes.append({
            "id": p["id"],
            "mpn": p["mpn"],
            "title": p["title"],
            "category": p["category"],
            "brand": p["brand"],
            "site": p["site"],
            "gender": p["gender"],
            "pack": p["pack"],
            "motivo": "Sem sugestões visuais válidas"
        })

# 👕 Agrupar variantes por mpn
mpn_variantes = defaultdict(list)
for p in produtos_validos:
    mpn_variantes[p["mpn"]].append({
        "id": p["id"],
        "size": p["size"],
        "availability": p["availability"]
    })

# 📝 Gerar JSON final
saida_json = []
vistos = set()

for produto in produtos_validos:
    mpn = produto["mpn"]
    if mpn in vistos:
        continue
    vistos.add(mpn)
    variantes = mpn_variantes.get(mpn, [])
    id_base = next((v["id"] for v in variantes if v["availability"] == "in stock"), variantes[0]["id"])
    
    saida_json.append({
        "id": id_base,
        "mpn": mpn,
        "title": produto["title"],
        "image": produto["image_link"],
        "gender": produto["gender"],
        "site": produto["site"],
        "category": produto["category"],
        "brand": produto["brand"],
        "price": produto.get("price", ""),
        "sale_price": produto.get("sale_price", ""),
        "variantes": variantes,
        "sugestoes": sugestoes_dict.get(id_base, [])
    })

saida_json_final = {
    "gerado_em": datetime.utcnow().isoformat(),
    "produtos": saida_json
}

with open("upselling_final.json", "w", encoding="utf-8") as f:
    json.dump(saida_json_final, f, ensure_ascii=False, indent=2)

with open("produtos_sem_sugestoes.json", "w", encoding="utf-8") as f:
    json.dump(produtos_sem_sugestoes, f, ensure_ascii=False, indent=2)

print("✅ JSON final criado: upselling_final.json")
print("📁 Log criado: produtos_sem_sugestoes.json")

# 📤 Upload para GitHub
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

# JSON principal
get_resp = requests.get(api_url, headers=headers)
sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
with open(filename, "rb") as f:
    content = base64.b64encode(f.read()).decode()
payload = {
    "message": "Atualizar upselling JSON",
    "content": content,
    "branch": branch
}
if sha:
    payload["sha"] = sha
put_resp = requests.put(api_url, headers=headers, json=payload)
if put_resp.status_code in [200, 201]:
    print("✅ JSON copiado para o GitHub com sucesso.")
else:
    print("❌ Erro ao enviar para o GitHub:", put_resp.json())

# Log
log_filename = "produtos_sem_sugestoes.json"
log_api_url = f"https://api.github.com/repos/{repo}/contents/{log_filename}"
with open(log_filename, "rb") as f:
    log_content = base64.b64encode(f.read()).decode()
log_get_resp = requests.get(log_api_url, headers=headers)
log_sha = log_get_resp.json().get("sha") if log_get_resp.status_code == 200 else None
log_payload = {
    "message": "Atualizar log de produtos sem sugestões",
    "content": log_content,
    "branch": branch
}
if log_sha:
    log_payload["sha"] = log_sha
log_put_resp = requests.put(log_api_url, headers=headers, json=log_payload)
if log_put_resp.status_code in [200, 201]:
    print("✅ Log de produtos sem sugestões enviado para o GitHub.")
else:
    print("❌ Erro ao enviar log para o GitHub:", log_put_resp.json())
