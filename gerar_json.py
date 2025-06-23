# gerar_json.py
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

# Configurações
xml_url = "https://www.bzronline.com/extend/catalog_24.xml"
image_folder = "imagens"
os.makedirs(image_folder, exist_ok=True)

# Carregar modelo ResNet50
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
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(img_url, headers=headers, timeout=10)
        if "image" not in r.headers.get("Content-Type", ""):
            print(f"❌ Ignorado (não é imagem): {img_url}")
            return None
        img = Image.open(BytesIO(r.content)).convert('RGB')
        img.save(image_path)

    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features.cpu().numpy().flatten()

# Ler XML
print("🔽 A descarregar XML...")
r = requests.get(xml_url)
tree = ET.ElementTree(ET.fromstring(r.content))
root = tree.getroot()
ns = {'g': 'http://base.google.com/ns/1.0', 'atom': 'http://www.w3.org/2005/Atom'}

produtos = []
for entry in root.findall('atom:entry', ns):
    try:
        produtos.append({
            "id": entry.find('g:id', ns).text,
            "title": entry.find('g:title', ns).text.strip(),
            "brand": entry.find('g:brand', ns).text.strip() if entry.find('g:brand', ns) is not None else "",
            "category": entry.find('g:category', ns).text or "",
            "image_link": entry.find('g:image_link', ns).text,
            "site": entry.find('g:site', ns).text if entry.find('g:site', ns) is not None else "",
            "gender": entry.find('g:gender', ns).text if entry.find('g:gender', ns) is not None else "",
        })
    except:
        continue

# Extrair embeddings visuais
print("🔍 A extrair embeddings visuais...")
embeddings = []
produtos_validos = []
for p in produtos:
    image_filename = f"{p['id']}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    emb = get_image_embedding(p["image_link"], image_path)
    if emb is not None:
        embeddings.append(emb)
        produtos_validos.append(p)

embeddings = np.array(embeddings)
similarity_matrix = cosine_similarity(embeddings)

# Criar sugestões
print("🧠 A gerar sugestões...")
sugestoes_dict = {}
for i, base in enumerate(produtos_validos):
    candidatos = [j for j, p in enumerate(produtos_validos)
                  if p["site"] == base["site"]
                  and p["gender"] == base["gender"]
                  and p["id"] != base["id"]
                  and p["category"].split(">")[-1].strip() != base["category"].split(">")[-1].strip()]

    if not candidatos:
        sugestoes_dict[base["id"]] = []
        continue

    sim_scores = similarity_matrix[i][candidatos]
    top_indices = np.argsort(sim_scores)[::-1][:5]
    sugestoes = [produtos_validos[candidatos[j]]["id"] for j in top_indices]
    sugestoes_dict[base["id"]] = sugestoes

# Gerar JSON
print("💾 A guardar JSON...")
output = []
for p in produtos_validos:
    output.append({
        "id": p["id"],
        "title": p["title"],
        "brand": p["brand"],
        "site": p["site"],
        "gender": p["gender"],
        "category": p["category"],
        "image_link": p["image_link"],
        "suggestions": sugestoes_dict.get(p["id"], [])
    })

with open("produtos_upselling.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ JSON criado com sucesso: produtos_upselling.json")
