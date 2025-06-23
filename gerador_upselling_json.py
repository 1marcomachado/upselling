
import os
import xml.etree.ElementTree as ET
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Caminhos
xml_url = "https://www.bzronline.com/extend/catalog_24.xml"
json_file = "sugestoes_upselling.json"

# Carregar XML da web
response = requests.get(xml_url)
tree = ET.ElementTree(ET.fromstring(response.content))
root = tree.getroot()

ns = {'g': 'http://base.google.com/ns/1.0', 'atom': 'http://www.w3.org/2005/Atom'}

# Extrair produtos
produtos = []
for entry in root.findall("atom:entry", ns):
    id_ = entry.find("g:id", ns)
    title = entry.find("g:title", ns)
    category = entry.find("g:category", ns)
    image = entry.find("g:image_link", ns)
    site = entry.find("g:site", ns)
    age = entry.find("g:age_group", ns)

    if None in (id_, title, category, image, site):
        continue

    produtos.append({
        "id": id_.text,
        "title": title.text.strip(),
        "category": category.text.strip(),
        "image": image.text.strip(),
        "site": site.text.strip(),
        "age_group": age.text.strip() if age is not None else "adult"
    })

# Gerar sugestões básicas com base em categoria e idade
sugestoes_dict = {}
for p in produtos:
    base_cat = p["category"].split(">")[-1].strip()
    base_site = p["site"]
    base_age = p["age_group"]

    candidatos = [
        s for s in produtos
        if s["id"] != p["id"]
        and s["site"] == base_site
        and s["age_group"] == base_age
        and s["category"].split(">")[-1].strip() != base_cat
    ]

    sugestoes = candidatos[:5]
    sugestoes_dict[p["id"]] = {
        "produto_base": p,
        "sugestoes": sugestoes
    }

# Guardar como JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(sugestoes_dict, f, indent=2, ensure_ascii=False)

print(f"✅ JSON gerado com {len(sugestoes_dict)} produtos: {json_file}")
