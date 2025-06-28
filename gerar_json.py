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

# Carrega variáveis do ambiente
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
repo = os.getenv("GITHUB_REPO")
branch = "main"
filename = "upselling_final.json"
log_filename = "produtos_sem_sugestoes.json"

api_url = f"https://api.github.com/repos/{repo}/contents/{filename}"
log_api_url = f"https://api.github.com/repos/{repo}/contents/{log_filename}"

xml_url = "https://www.bzronline.com/extend/catalog_24.xml"
image_folder = "imagens"
os.makedirs(image_folder, exist_ok=True)

categorias_complementares = {
    "T-shirts": ["Calções", "Hoodies"],
    "Calções": ["T-shirts", "Hoodies"],
    "Hoodies": ["T-shirts", "Calções"],
    "Calças": ["Camisolas", "Polos", "T-shirts"],
    "Óculos de Natação": ["Chinelos", "Toucas"],
    "Chinelos": ["Óculos de Natação", "Toucas"],
}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

print("🔽 A descarregar XML...")
r = session.get(xml_url, timeout=30)
r.raise_for_status()
tree = ET.ElementTree(ET.fromstring(r.content))
root = tree.getroot()

ns = {'g': 'http://base.google.com/ns/1.0', 'atom': 'http://www.w3.org/2005/Atom'}
produtos = []

for entry in root.findall('atom:entry', ns):
    produto = {
        "id": entry.findtext('g:id', default="", namespaces=ns),
        "title": entry.findtext('g:title', default="", namespaces=ns).strip(),
        "category": entry.findtext('g:category', default="", namespaces=ns),
        "brand": entry.findtext('g:brand', default="", namespaces=ns).strip(),
        "image_link": entry.findtext('g:image_link', default="", namespaces=ns),
        "site": entry.findtext('g:site', default="", namespaces=ns),
        "gender": entry.findtext('g:gender', default="", namespaces=ns),
        "pack": entry.findtext('g:pack', default="", namespaces=ns),
        "price": entry.findtext('g:price', default="", namespaces=ns),
        "sale_price": entry.findtext('g:sale_price', default="", namespaces=ns)
    }
    produtos.append(produto)

# Modelo e transformações
print("🧠 A preparar embeddings com cache...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval().to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_embedding(img_url, image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
    else:
        resp = session.get(img_url, timeout=10)
        if "image" not in resp.headers.get("Content-Type", ""):
            return None
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        img.save(image_path)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features.cpu().numpy().flatten()

embeddings_file = "embeddings.npy"
ids_file = "ids_embeddings.json"
embeddings = []
ids_existentes = []
if os.path.exists(embeddings_file) and os.path.exists(ids_file):
    embeddings = np.load(embeddings_file)
    with open(ids_file, "r", encoding="utf-8") as f:
        ids_existentes = json.load(f)

produtos_validos = []
novos_embeddings = []
for p in produtos:
    if p["id"] in ids_existentes:
        produtos_validos.append(p)
        continue
    image_path = os.path.join(image_folder, f"{p['id']}.jpg")
    emb = get_image_embedding(p["image_link"], image_path)
    if emb is not None:
        novos_embeddings.append(emb)
        produtos_validos.append(p)
        ids_existentes.append(p["id"])

if novos_embeddings:
    embeddings = np.vstack((embeddings, novos_embeddings)) if len(embeddings) else np.array(novos_embeddings)
    np.save(embeddings_file, embeddings)
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(ids_existentes, f, indent=2, ensure_ascii=False)

print("📊 A calcular similaridades...")
similarity_matrix = cosine_similarity(embeddings)
sugestoes_dict = {}
produtos_sem_sugestoes = []

for i, produto in enumerate(produtos_validos):
    last = produto["category"].split(">")[-1].strip()
    second = produto["category"].split(">")[1].strip() if len(produto["category"].split(">")) > 1 else ""

    candidatos = [j for j, p in enumerate(produtos_validos)
                  if p["site"] == produto["site"] and p["brand"] == produto["brand"]
                  and p["gender"] == produto["gender"] and p["id"] != produto["id"]]

    if produto["site"] == "2":
        if produto["pack"]:
            candidatos = [j for j in candidatos if produtos_validos[j]["pack"] == produto["pack"]]
        elif second:
            candidatos = [j for j in candidatos if len(produtos_validos[j]["category"].split(">")) > 1 and
                          produtos_validos[j]["category"].split(">")[1].strip() == second and
                          produtos_validos[j]["category"].split(">")[-1].strip() != last]
        else:
            candidatos = [j for j in candidatos if produtos_validos[j]["category"].split(">")[-1].strip() != last]
    else:
        candidatos = [j for j in candidatos if produtos_validos[j]["category"].split(">")[-1].strip() != last]

    if last in categorias_complementares:
        candidatos = [j for j in candidatos if produtos_validos[j]["category"].split(">")[-1].strip() in categorias_complementares[last]]

    if candidatos:
        scores = similarity_matrix[i][candidatos]
        top_idx = np.argsort(scores)[::-1][:5]
        sugestoes_dict[produto["id"]] = [produtos_validos[candidatos[j]]["id"] for j in top_idx]
    else:
        produtos_sem_sugestoes.append({
            "id": produto["id"],
            "title": produto["title"],
            "category": produto["category"],
            "brand": produto["brand"],
            "gender": produto["gender"],
            "site": produto["site"],
            "motivo": "Sem candidatos compatíveis"
        })

saida_json = [{**p, "sugestoes": sugestoes_dict.get(p["id"], [])} for p in produtos_validos]
saida_json_final = {"gerado_em": datetime.utcnow().isoformat(), "produtos": saida_json}

with open(filename, "w", encoding="utf-8") as f:
    json.dump(saida_json_final, f, indent=2, ensure_ascii=False)
with open(log_filename, "w", encoding="utf-8") as f:
    json.dump(produtos_sem_sugestoes, f, indent=2, ensure_ascii=False)

print("✅ JSON final criado: upselling_final.json")
print(f"📋 Produtos sem sugestões: {len(produtos_sem_sugestoes)}")

# Upload JSON principal
headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
resp = requests.get(api_url, headers=headers)
sha = resp.json().get("sha") if resp.status_code == 200 else None

with open(filename, "rb") as f:
    content = base64.b64encode(f.read()).decode()
payload = {"message": "Atualizar upselling JSON", "content": content, "branch": branch}
if sha: payload["sha"] = sha
put = requests.put(api_url, headers=headers, json=payload)
print("✅ JSON enviado" if put.status_code in [200, 201] else f"❌ Erro: {put.json()}")

# Upload log
log_resp = requests.get(log_api_url, headers=headers)
log_sha = log_resp.json().get("sha") if log_resp.status_code == 200 else None
with open(log_filename, "rb") as f:
    log_content = base64.b64encode(f.read()).decode()
log_payload = {"message": "Atualizar log de produtos sem sugestões", "content": log_content, "branch": branch}
if log_sha: log_payload["sha"] = log_sha
log_put = requests.put(log_api_url, headers=headers, json=log_payload)
print("✅ Log enviado" if log_put.status_code in [200, 201] else f"❌ Erro log: {log_put.json()}")
