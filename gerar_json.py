import xml.etree.ElementTree as ET
import pandas as pd
import requests
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

print("🔽 A descarregar XML...")
r = requests.get(xml_url)
tree = ET.ElementTree(ET.fromstring(r.content))
root = tree.getroot()

ns = {'g': 'http://base.google.com/ns/1.0', 'atom': 'http://www.w3.org/2005/Atom'}
produtos = []

for entry in root.findall('atom:entry', ns):
    id_ = entry.find('g:id', ns).text if entry.find('g:id', ns) is not None else ""
    title = entry.find('g:title', ns).text.strip() if entry.find('g:title', ns) is not None else ""
    category = entry.find('g:category', ns).text if entry.find('g:category', ns) is not None else ""
    brand = entry.find('g:brand', ns).text.strip() if entry.find('g:brand', ns) is not None else ""
    image_link = entry.find('g:image_link', ns).text if entry.find('g:image_link', ns) is not None else ""
    site = entry.find('g:site', ns).text if entry.find('g:site', ns) is not None else ""
    gender = entry.find('g:gender', ns).text if entry.find('g:gender', ns) is not None else ""
    pack = entry.find('g:pack', ns).text if entry.find('g:pack', ns) is not None else ""

    produtos.append({
        "id": id_,
        "title": title,
        "category": category,
        "brand": brand,
        "image_link": image_link,
        "site": site,
        "gender": gender,
        "pack": pack
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_image_embedding(img_url, image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
    else:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
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

print("🧠 A preparar embeddings com cache...")
embeddings_file = "embeddings.npy"
ids_file = "ids_embeddings.json"

if os.path.exists(embeddings_file) and os.path.exists(ids_file):
    embeddings = np.load(embeddings_file)
    with open(ids_file, "r", encoding="utf-8") as f:
        ids_existentes = json.load(f)
else:
    embeddings = []
    ids_existentes = []

produtos_validos = []
novos_embeddings = []

for p in produtos:
    if p["id"] in ids_existentes:
        produtos_validos.append(p)
        continue
    image_filename = f"{p['id']}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    emb = get_image_embedding(p["image_link"], image_path)
    if emb is not None:
        novos_embeddings.append(emb)
        produtos_validos.append(p)
        ids_existentes.append(p["id"])

if novos_embeddings:
    if isinstance(embeddings, list) or len(embeddings) == 0:
        embeddings = np.array(novos_embeddings)
    else:
        embeddings = np.vstack((embeddings, novos_embeddings))
    np.save(embeddings_file, embeddings)
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(ids_existentes, f, ensure_ascii=False, indent=2)

print("📊 A calcular similaridades...")
similarity_matrix = cosine_similarity(embeddings)

sugestoes_dict = {}
for i, produto in enumerate(produtos_validos):
    last_level_base = produto["category"].split(">")[-1].strip()
    second_level_base = produto["category"].split(">")[1].strip() if len(produto["category"].split(">")) > 1 else ""

    candidatos = [j for j, p in enumerate(produtos_validos)
                   if p["site"] == produto["site"] and
                      p["brand"] == produto["brand"] and
                      p["gender"] == produto["gender"] and
                      p["id"] != produto["id"]]

    if produto["site"] == "2":
        if produto["pack"]:
            candidatos = [j for j in candidatos if produtos_validos[j]["pack"] == produto["pack"]]
        elif second_level_base:
            candidatos = [j for j in candidatos
                           if len(produtos_validos[j]["category"].split(">")) > 1 and
                              produtos_validos[j]["category"].split(">")[1].strip() == second_level_base and
                              produtos_validos[j]["category"].split(">")[-1].strip() != last_level_base]
        else:
            candidatos = [j for j in candidatos if produtos_validos[j]["category"].split(">")[-1].strip() != last_level_base]
    else:
        candidatos = [j for j in candidatos if produtos_validos[j]["category"].split(">")[-1].strip() != last_level_base]

    if last_level_base in categorias_complementares:
        candidatos = [j for j in candidatos if produtos_validos[j]["category"].split(">")[-1].strip() in categorias_complementares[last_level_base]]

    if candidatos:
        sim_scores = similarity_matrix[i][candidatos]
        indices_ordenados = np.argsort(sim_scores)[::-1][:5]
        sugestoes = [produtos_validos[candidatos[j]]["id"] for j in indices_ordenados]
        sugestoes_dict[produto["id"]] = sugestoes

# Salva para JSON
saida_json = []
for produto in produtos_validos:
    saida_json.append({
        "id": produto["id"],
        "title": produto["title"],
        "image": produto["image_link"],
        "gender": produto["gender"],
        "site": produto["site"],
        "category": produto["category"],
        "brand": produto["brand"],
        "sugestoes": sugestoes_dict.get(produto["id"], [])
    })

saida_json_final = {
    "gerado_em": datetime.utcnow().isoformat(),
    "produtos": saida_json
}

with open("upselling_final.json", "w", encoding="utf-8") as f:
    json.dump(saida_json_final, f, ensure_ascii=False, indent=2)

print("✅ JSON final criado: upselling_final.json")
