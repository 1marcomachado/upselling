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

# =========================
# ⚙️ Parâmetros ajustáveis
# =========================
TOTAL_LIMIT = 30        # nº máximo de sugestões por produto
DIVERSITY_RATIO = 0.20  # fração mínima dedicada a diversidade (≈1 por categoria até 20% do total)
SOFT_CAP_RATIO = 0.40   # teto "soft" por categoria (40% do total)
# =========================

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

# 📦 Parse produtos (sem title “genérico”)
produtos = []
for entry in root.findall('atom:entry', ns):
    id_ = entry.find('g:id', ns).text if entry.find('g:id', ns) is not None else ""
    mpn = entry.find('g:mpn', ns).text if entry.find('g:mpn', ns) is not None else ""
    title_pt = entry.find('g:title_pt', ns).text.strip() if entry.find('g:title_pt', ns) is not None else ""
    title_es = entry.find('g:title_es', ns).text.strip() if entry.find('g:title_es', ns) is not None else ""
    title_en = entry.find('g:title_en', ns).text.strip() if entry.find('g:title_en', ns) is not None else ""
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
        "title_pt": title_pt,
        "title_es": title_es,
        "title_en": title_en,
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

# Helpers para categorias / exceções
def _split_cat(cat):
    return [c.strip() for c in cat.split(">")] if cat else []

def _first_level(cat):
    parts = _split_cat(cat)
    return parts[0].casefold() if parts else ""

def _is_accessories(cat_str):
    return _first_level(cat_str) == "acessórios" if cat_str else False

# 🧠 Preparar embeddings com cache
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

# 🔄 Construir sugestões (prioriza complementares; Acessórios sem brand; mantém sempre site+gender)
sugestoes_dict = {}
produtos_sem_sugestoes = []

for p in produtos_validos:
    base_mpn = p["mpn"]
    if base_mpn not in mpn_to_embedding:
        continue

    i = mpn_list.index(base_mpn)

    site_base   = (p.get("site") or "").strip()
    brand_base  = (p.get("brand") or "").strip()
    gender_base = (p.get("gender") or "").strip()
    cat_full    = (p.get("category") or "").strip()

    # 1) categorias complementares (ordem de preferência)
    categorias_validas = None
    if cat_full in categorias_complementares:
        categorias_validas = [c.strip() for c in categorias_complementares[cat_full] if c and c.strip()]

    acessorios_em_validas = False
    if categorias_validas:
        acessorios_em_validas = any(_is_accessories(c) for c in categorias_validas)

    # 2) construir pools (sempre MESMO site + MESMO gender)
    cand_preferidos = []  # dentro das complementares
    cand_fallback  = []   # fora das complementares

    for j, mpn_cand in enumerate(mpn_list):
        if mpn_cand == base_mpn:
            continue
        cand = mpn_to_produto[mpn_cand]

        if (cand.get("site") or "").strip() != site_base:
            continue
        if (cand.get("gender") or "").strip() != gender_base:
            continue

        cand_cat   = (cand.get("category") or "").strip()
        cand_brand = (cand.get("brand") or "").strip()

        in_valid = bool(categorias_validas) and (cand_cat in categorias_validas)
        is_acess = _is_accessories(cand_cat)

        # regra brand:
        # - se categoria é Acessórios e está nas complementares → brand NÃO é exigida
        # - nos restantes casos → brand tem de coincidir
        if in_valid and is_acess:
            brand_ok = True
        else:
            #brand_ok = (cand_brand == brand_base)
            brand_ok = True

        if not brand_ok:
            # ainda pode ir para fallback? (mantemos a mesma exceção para Acessórios)
            if is_acess and acessorios_em_validas:
                # permitir sem brand também no fallback para manter coerência
                pass
            else:
                continue

        if in_valid:
            cand_preferidos.append(j)
        else:
            cand_fallback.append(j)

    total_pool = len(cand_preferidos) + len(cand_fallback)
    if total_pool == 0:
        produtos_sem_sugestoes.append({
            "id": p["id"],
            "mpn": p["mpn"],
            "title": {
                "pt": p["title_pt"],
                "es": p["title_es"],
                "en": p["title_en"]
            },
            "category": p["category"],
            "brand": p["brand"],
            "site": p["site"],
            "gender": p["gender"],
            "pack": p["pack"],
            "motivo": "Sem candidatos (site/gender/brand + regra Acessórios)"
        })
        continue

    # 3) ranking + diversidade
    diversity_quota = max(1, int(TOTAL_LIMIT * DIVERSITY_RATIO))
    max_per_cat     = max(2, int(TOTAL_LIMIT * SOFT_CAP_RATIO))

    def rank_by_cat(indices):
        cat_to_candidates = defaultdict(list)
        for j in indices:
            cand = mpn_to_produto[mpn_list[j]]
            cat  = (cand.get("category") or "").strip()
            score = similarity_matrix[i][j]
            cat_to_candidates[cat].append((j, score))
        for cat in cat_to_candidates:
            cat_to_candidates[cat].sort(key=lambda x: x[1], reverse=True)

        cats_presentes = list(cat_to_candidates.keys())
        if categorias_validas:
            preferidas = [c for c in categorias_validas if c in cat_to_candidates]
            restantes  = [c for c in cats_presentes if c not in set(preferidas)]
            restantes.sort(key=lambda c: cat_to_candidates[c][0][1], reverse=True)
            cats_ordenadas = preferidas + restantes
        else:
            cats_ordenadas = sorted(cats_presentes, key=lambda c: cat_to_candidates[c][0][1], reverse=True)
        return cats_ordenadas, cat_to_candidates

    # Fase 1: só complementares
    sugestoes_indices = []
    usados = set()
    cat_counts = defaultdict(int)

    cats_pref, map_pref = rank_by_cat(cand_preferidos)

    # diversidade mínima dentro das complementares
    for cat in cats_pref:
        if len(sugestoes_indices) >= diversity_quota:
            break
        top_idx, _ = map_pref[cat][0]
        if top_idx not in usados:
            sugestoes_indices.append(top_idx)
            usados.add(top_idx)
            cat_counts[cat] += 1

    # pool global preferidos (prioridade por categoria complementar e score)
    def is_cat_valid(c):
        return bool(categorias_validas) and (c in categorias_validas)

    pool_pref_global = sorted(
        [j for j in cand_preferidos if j not in usados],
        key=lambda x: (0 if is_cat_valid((mpn_to_produto[mpn_list[x]]["category"] or "").strip()) else 1,
                       -similarity_matrix[i][x])
    )

    for j in pool_pref_global:
        if len(sugestoes_indices) >= TOTAL_LIMIT:
            break
        cat = (mpn_to_produto[mpn_list[j]]["category"] or "").strip()
        if cat_counts[cat] < max_per_cat:
            sugestoes_indices.append(j)
            usados.add(j)
            cat_counts[cat] += 1

    # Fase 2: fallback (fora das complementares)
    if len(sugestoes_indices) < TOTAL_LIMIT and cand_fallback:
        cats_fb, map_fb = rank_by_cat(cand_fallback)

        # completar diversidade mínima se necessário
        if len(sugestoes_indices) < diversity_quota:
            for cat in cats_fb:
                if len(sugestoes_indices) >= diversity_quota:
                    break
                top_idx, _ = map_fb[cat][0]
                if top_idx not in usados:
                    sugestoes_indices.append(top_idx)
                    usados.add(top_idx)
                    cat_counts[cat] += 1

        pool_fb_global = sorted(
            [j for j in cand_fallback if j not in usados],
            key=lambda x: -similarity_matrix[i][x]
        )

        for j in pool_fb_global:
            if len(sugestoes_indices) >= TOTAL_LIMIT:
                break
            cat = (mpn_to_produto[mpn_list[j]]["category"] or "").strip()
            if cat_counts[cat] < max_per_cat:
                sugestoes_indices.append(j)
                usados.add(j)
                cat_counts[cat] += 1

        # relaxar teto por categoria se ainda faltar
        if len(sugestoes_indices) < TOTAL_LIMIT:
            restantes_relax = [j for j in pool_fb_global if j not in usados]
            for j in restantes_relax:
                if len(sugestoes_indices) >= TOTAL_LIMIT:
                    break
                sugestoes_indices.append(j)
                usados.add(j)

    # limitar ao universo existente
    all_candidates = set(cand_preferidos) | set(cand_fallback)
    sugestoes_indices = [idx for idx in sugestoes_indices if idx in all_candidates][:min(TOTAL_LIMIT, len(all_candidates))]

    sugestoes = [mpn_list[idx] for idx in sugestoes_indices]
    sugestoes_dict[p["id"]] = sugestoes

# 👕 Agrupar variantes por mpn
mpn_variantes = defaultdict(list)
for p in produtos_validos:
    mpn_variantes[p["mpn"]].append({
        "id": p["id"],
        "size": p["size"],
        "availability": p["availability"]
    })

# 📝 Gerar JSON final (mostrar todas as variantes, mas só incluir produtos com >=1 variante em stock)
saida_json = []
vistos = set()

def _is_instock(av):
    return ((av or "").strip().lower() == "in stock")

for produto in produtos_validos:
    mpn = produto["mpn"]
    if mpn in vistos:
        continue
    vistos.add(mpn)

    variantes_all = mpn_variantes.get(mpn, [])

    # 🔍 verifica se há pelo menos uma variante com stock
    variantes_instock = [v for v in variantes_all if _is_instock(v.get("availability"))]
    if not variantes_instock:
        # ❌ sem stock em qualquer variante → não incluir no JSON final
        continue

    # ✅ escolher id_base de uma variante com stock
    id_base = variantes_instock[0]["id"]

    saida_json.append({
        "id": id_base,
        "mpn": mpn,
        "title": {
            "pt": produto.get("title_pt", ""),
            "es": produto.get("title_es", ""),
            "en": produto.get("title_en", "")
        },
        "image": produto["image_link"],
        "gender": produto["gender"],
        "site": produto["site"],
        "category": produto["category"],
        "brand": produto["brand"],
        "price": produto.get("price", ""),
        "sale_price": produto.get("sale_price", ""),
        # ✅ expor TODAS as variantes (com e sem stock)
        "variantes": variantes_all,
        # manter lookup de sugestões pelo id_base (em stock)
        "sugestoes": sugestoes_dict.get(id_base, sugestoes_dict.get(produto["id"], []))
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
    try:
        print("❌ Erro ao enviar para o GitHub:", put_resp.json())
    except Exception:
        print("❌ Erro ao enviar para o GitHub (sem JSON)")

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
    try:
        print("❌ Erro ao enviar log para o GitHub:", log_put_resp.json())
    except Exception:
        print("❌ Erro ao enviar log para o GitHub (sem JSON)")
