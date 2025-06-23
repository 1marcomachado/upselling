import os
import json
import xml.etree.ElementTree as ET
import requests
from dotenv import load_dotenv

load_dotenv()

xml_url = os.getenv("XML_FEED_URL", "https://www.bzronline.com/extend/catalog_24.xml")
xml_filename = "catalog_24.xml"
json_output = "produtos_upselling.json"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(xml_url, headers=headers)
response.raise_for_status()

with open(xml_filename, "wb") as f:
    f.write(response.content)

tree = ET.parse(xml_filename)
root = tree.getroot()

ns = {
    'g': 'http://base.google.com/ns/1.0',
    'atom': 'http://www.w3.org/2005/Atom'
}

produtos = []

for entry in root.findall('atom:entry', ns):
    produto = {
        "id": entry.findtext('g:id', default="", namespaces=ns),
        "title_pt": entry.findtext('g:title_pt', default="", namespaces=ns).strip(),
        "title_en": entry.findtext('g:title_en', default="", namespaces=ns).strip(),
        "title_es": entry.findtext('g:title_es', default="", namespaces=ns).strip(),
        "category": entry.findtext('g:category', default="", namespaces=ns),
        "brand": entry.findtext('g:brand', default="", namespaces=ns).strip(),
        "image_link": entry.findtext('g:image_link', default="", namespaces=ns),
        "site": entry.findtext('g:site', default="", namespaces=ns),
        "gender": entry.findtext('g:gender', default="", namespaces=ns),
        "pack": entry.findtext('g:pack', default="", namespaces=ns),
    }
    produtos.append(produto)

with open(json_output, "w", encoding="utf-8") as f:
    json.dump(produtos, f, ensure_ascii=False, indent=2)

print(f"✅ JSON criado com sucesso: {json_output}")
