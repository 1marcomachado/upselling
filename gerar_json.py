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
from datetime import datetime, timezone
import base64
from dotenv import load_dotenv
from collections import defaultdict
import math
import re

# =========================
# ⚙️ Parâmetros ajustáveis
# =========================
TOTAL_LIMIT = 30        # nº máximo de sugestões por produto
DIVERSITY_RATIO = 0.20  # fração mínima dedicada a diversidade (≈1 por categoria até 20% do total)
SOFT_CAP_RATIO = 0.40   # teto "soft" por categoria (40% do total)

# =========================
# ⚖️ Pesos para o ranking (sem +vendidos)
# =========================
W_STOCK = 0.12      # 12%
W_NEW   = 0.39      # 39%
W_SIM   = max(0.0, 1.0 - (W_STOCK + W_NEW))  # resto vai para similaridade

NOVELTY_HALF_LIFE_DAYS = 180
DEFAULT_NEWNESS_WHEN_MISSING = 0.0

# =========================
# 🔐 Env / Endpoints
# =========================
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

# Utils
def _clamp01(x):
    try:
        return 0.0 if x is None else max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

# =========================
# 🔽 Descarregar XML
# =========================
print("🔽 A descarregar XML...")
r = session.get(xml_url, timeout=30)
r.raise_for_status()
xml_content = r.content
tree = ET.ElementTree(ET.fromstring(xml_content))
root = tree.getroot()
ns = {'g': 'http://base.google.com/ns/1.0', 'atom': 'http://www.w3.org/2005/Atom'}

def _sanitize_date_raw(s: str) -> str:
    s = (s or "").strip()
    bads = {"0000-00-00", "0000/00/00", "0000-00-00T00:00:00", "0000-00-00 00:00:00"}
    if not s or s in bads or s.startswith("0000-00-00"):
        return ""
    return s

def _parse_date_guess(s: str):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    if re.match(r"^0{4}[-/ ]?0{2}[-/ ]?0{2}", s):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.year < 1970 or dt > datetime.now(timezone.utc):
                return None
            return dt
        except Exception:
            continue
    return None

# =========================
# 📦 Parse produtos
# =========================
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

    # Novos campos
    color = entry.find('g:color', ns).text.strip() if entry.find('g:color', ns) is not None else ""
    age_group_raw = entry.find('g:age_group', ns).text if entry.find('g:age_group', ns) is not None else ""
    age_group = (age_group_raw or "").strip()
    modalidade = entry.find('g:modalidade', ns).text if entry.find('g:modalidade', ns) is not None else ""

    # Novidade — usar só release_date
    release_txt = entry.find('g:release_date', ns).text if entry.find('g:release_date', ns) is not None else ""
    release_txt = _sanitize_date_raw(release_txt)

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
        "availability": availability,
        "color": color,
        "age_group": age_group,
        "modalidade": modalidade,
        "updated_raw": release_txt  # só release_date
    })

# =========================
# 🧠 ResNet50 para embeddings
# =========================
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

# =========================
# Helpers categorias / regras
# =========================
def _split_cat(cat):
    return [c.strip() for c in cat.split(">")] if cat else []

def _first_level(cat):
    parts = _split_cat(cat)
    return parts[0].casefold() if parts else ""

def _is_accessories(cat_str):
    return _first_level(cat_str) == "acessórios" if cat_str else False

def _norm_age_group(s: str) -> str:
    s = (s or "").strip().lower()
    if "adulto" in s and "crian" in s:
        return "adulto e criança"
    if "adulto" in s:
        return "adulto"
    if "crian" in s:
        return "criança"
    return ""

def _age_compat(base_age: str, cand_age: str) -> bool:
    b = _norm_age_group(base_age)
    c = _norm_age_group(cand_age)
    if not b or not c:
        return False
    if b == c:
        return True
    return b == "adulto e criança" or c == "adulto e criança"

def _modalidade_compat(base_mod: str, cand_mod: str) -> bool:
    bm = (base_mod or "").strip().lower()
    cm = (cand_mod or "").strip().lower()
    if bm and cm:
        return bm == cm
    return True

def _is_instock(av):
    return ((av or "").strip().lower() == "in stock")

# =========================
# 🧠 Preparar embeddings com cache
# =========================
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

for i, mpn in enumerate(mpns_existentes):
    mpn_to_embedding[mpn] = embeddings[i]

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

if novos_embeddings:
    novos_embeddings = np.array(novos_embeddings)
    embeddings = np.vstack((embeddings, novos_embeddings)) if len(embeddings) else novos_embeddings
    np.save(embeddings_file, embeddings)
    with open(mpns_file, "w", encoding="utf-8") as f:
        json.dump(list(mpns_adicionados), f, ensure_ascii=False, indent=2)

produtos_validos = []
for p in produtos:
    mpn = p["mpn"]
    if mpn in mpn_to_embedding:
        produtos_validos.append(p)
        mpn_to_produto[mpn] = p

print(f"✅ Produtos com embeddings: {len(produtos_validos)}")

# =========================
# 📊 Sinais: STOCK / NOVIDADE
# =========================
mpn_stock_ratio = defaultdict(float)
mpn_var_counts  = defaultdict(int)
mpn_var_instk   = defaultdict(int)

for p in produtos_validos:
    mpn = p["mpn"]
    mpn_var_counts[mpn] += 1
    if _is_instock(p.get("availability")):
        mpn_var_instk[mpn] += 1

for mpn in mpn_var_counts:
    total = mpn_var_counts[mpn]
    instk = mpn_var_instk[mpn]
    mpn_stock_ratio[mpn] = (instk / total) if total > 0 else 0.0

now = datetime.now(timezone.utc)
mpn_best_dt = {}
for p in produtos_validos:
    mpn = p["mpn"]
    dt = _parse_date_guess(p.get("updated_raw", ""))
    if dt and (mpn not in mpn_best_dt or dt > mpn_best_dt[mpn]):
        mpn_best_dt[mpn] = dt

mpn_novidade_score = defaultdict(float)
for mpn, dt in mpn_best_dt.items():
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    freshness = max(0.0, 1.0 - (days / NOVELTY_HALF_LIFE_DAYS))
    mpn_novidade_score[mpn] = freshness

# =========================
# 📊 Similaridade visual
# =========================
print("📊 A calcular similaridades...")
mpn_list = list(mpn_to_embedding.keys())
mpn_embeddings = np.array([mpn_to_embedding[m] for m in mpn_list])
similarity_matrix = cosine_similarity(mpn_embeddings)

def _combined_score(i_idx: int, j_idx: int) -> float:
    sim = float(similarity_matrix[i_idx][j_idx])
    cand_mpn = mpn_list[j_idx]
    stock = mpn_stock_ratio.get(cand_mpn, 0.0)
    newn  = mpn_novidade_score.get(cand_mpn, DEFAULT_NEWNESS_WHEN_MISSING)
    return float(W_SIM * _clamp01(sim) + W_STOCK * _clamp01(stock) + W_NEW * _clamp01(newn))

# =========================
# 🔄 Construir sugestões
# =========================
sugestoes_dict = {}
produtos_sem_sugestoes = []

for p in produtos_validos:
    base_mpn = p["mpn"]
    if base_mpn not in mpn_to_embedding:
        continue

    i = mpn_list.index(base_mpn)

    site_base   = (p.get("site") or "").strip()
    gender_base = (p.get("gender") or "").strip()
    cat_full    = (p.get("category") or "").strip()

    categorias_validas = None
    if cat_full in categorias_complementares:
        categorias_validas = [c.strip() for c in categorias_complementares[cat_full] if c and c.strip()]

    # ——————————————————
    # 1) Universo elegível (filtros obrigatórios)
    # ——————————————————
    cand_all = []
    for j, mpn_cand in enumerate(mpn_list):
        if mpn_cand == base_mpn:
            continue
        cand = mpn_to_produto[mpn_cand]

        if (cand.get("site") or "").strip() != site_base:
            continue
        if (cand.get("gender") or "").strip() != gender_base:
            continue
        if not _age_compat(p.get("age_group"), cand.get("age_group")):
            continue
        if not _modalidade_compat(p.get("modalidade"), cand.get("modalidade")):
            continue

        cand_all.append(j)

    if not cand_all:
        produtos_sem_sugestoes.append({
            "id": p["id"],
            "mpn": p["mpn"],
            "title": {"pt": p["title_pt"], "es": p["title_es"], "en": p["title_en"]},
            "category": p["category"], "brand": p["brand"], "site": p["site"],
            "gender": p["gender"], "pack": p["pack"],
            "age_group": p.get("age_group", ""), "modalidade": p.get("modalidade", ""),
            "motivo": "Sem candidatos após filtros obrigatórios (site/gender/age_group/modalidade)"
        })
        continue

    # ——————————————————
    # 2) Partição: complementares vs fallback
    # ——————————————————
    cand_preferidos, cand_fallback = [], []
    for j in cand_all:
        cand = mpn_to_produto[mpn_list[j]]
        cand_cat = (cand.get("category") or "").strip()
        in_valid = bool(categorias_validas) and (cand_cat in categorias_validas)
        if in_valid:
            cand_preferidos.append(j)
        else:
            cand_fallback.append(j)

    diversity_quota = max(1, int(TOTAL_LIMIT * DIVERSITY_RATIO))
    max_per_cat     = max(2, int(TOTAL_LIMIT * SOFT_CAP_RATIO))

    def rank_by_cat(indices):
        cat_to_candidates = defaultdict(list)
        for j in indices:
            cand = mpn_to_produto[mpn_list[j]]
            cat  = (cand.get("category") or "").strip()
            score = _combined_score(i, j)
            cat_to_candidates[cat].append((j, score))
        for cat in cat_to_candidates:
            cat_to_candidates[cat].sort(key=lambda x: x[1], reverse=True)
        cats = list(cat_to_candidates.keys())
        if categorias_validas:
            preferidas = [c for c in categorias_validas if c in cat_to_candidates]
            restantes  = [c for c in cats if c not in set(preferidas)]
            restantes.sort(key=lambda c: cat_to_candidates[c][0][1], reverse=True)
            cats_ordenadas = preferidas + restantes
        else:
            cats_ordenadas = sorted(cats, key=lambda c: cat_to_candidates[c][0][1], reverse=True)
        return cats_ordenadas, cat_to_candidates

    sugestoes_indices, usados = [], set()
    cat_counts = defaultdict(int)

    # FASE A: Complementares primeiro
    cats_pref, map_pref = rank_by_cat(cand_preferidos)

    # A1) diversidade mínima entre complementares
    for cat in cats_pref:
        if len(sugestoes_indices) >= diversity_quota:
            break
        top_idx, _ = map_pref[cat][0]
        if top_idx not in usados:
            sugestoes_indices.append(top_idx); usados.add(top_idx); cat_counts[cat] += 1

    # A2) preencher até TOTAL_LIMIT com complementares (teto por categoria)
    pool_pref_global = sorted([j for j in cand_preferidos if j not in usados],
                              key=lambda x: -_combined_score(i, x))
    for j in pool_pref_global:
        if len(sugestoes_indices) >= TOTAL_LIMIT: break
        cat = (mpn_to_produto[mpn_list[j]]["category"] or "").strip()
        if cat_counts[cat] < max_per_cat:
            sugestoes_indices.append(j); usados.add(j); cat_counts[cat] += 1

    # A3) relaxar teto dentro das complementares se ainda faltar
    if len(sugestoes_indices) < TOTAL_LIMIT:
        for j in pool_pref_global:
            if len(sugestoes_indices) >= TOTAL_LIMIT: break
            if j in usados: continue
            sugestoes_indices.append(j); usados.add(j)

    # FASE B: Fallback (só se ainda faltar)
    if len(sugestoes_indices) < TOTAL_LIMIT and cand_fallback:
        cats_fb, map_fb = rank_by_cat(cand_fallback)

        # B1) completar diversidade mínima se necessário
        if len(sugestoes_indices) < diversity_quota:
            for cat in cats_fb:
                if len(sugestoes_indices) >= diversity_quota: break
                top_idx, _ = map_fb[cat][0]
                if top_idx not in usados:
                    sugestoes_indices.append(top_idx); usados.add(top_idx); cat_counts[cat] += 1

        # B2) preencher até TOTAL_LIMIT com fallback (teto por categoria)
        pool_fb_global = sorted([j for j in cand_fallback if j not in usados],
                                key=lambda x: -_combined_score(i, x))
        for j in pool_fb_global:
            if len(sugestoes_indices) >= TOTAL_LIMIT: break
            cat = (mpn_to_produto[mpn_list[j]]["category"] or "").strip()
            if cat_counts[cat] < max_per_cat:
                sugestoes_indices.append(j); usados.add(j); cat_counts[cat] += 1

        # B3) relaxar teto no fallback se ainda faltar
        if len(sugestoes_indices) < TOTAL_LIMIT:
            for j in pool_fb_global:
                if len(sugestoes_indices) >= TOTAL_LIMIT: break
                if j in usados: continue
                sugestoes_indices.append(j); usados.add(j)

    # cortar ao universo e ao TOTAL_LIMIT
    all_candidates = set(cand_all)
    sugestoes_indices = [idx for idx in sugestoes_indices if idx in all_candidates][:min(TOTAL_LIMIT, len(all_candidates))]

    sugestoes = [mpn_list[idx] for idx in sugestoes_indices]
    sugestoes_dict[p["id"]] = sugestoes

# =========================
# 👕 Agrupar variantes por mpn
# =========================
mpn_variantes = defaultdict(list)
for p in produtos_validos:
    mpn_variantes[p["mpn"]].append({
        "id": p["id"],
        "size": p["size"],
        "availability": p["availability"]
    })

# =========================
# 📝 JSON final (igual ao último, sem 'bestseller')
# =========================
saida_json = []
vistos = set()

for produto in produtos_validos:
    mpn = produto["mpn"]
    if mpn in vistos:
        continue
    vistos.add(mpn)

    variantes_all = mpn_variantes.get(mpn, [])
    variantes_instock = [v for v in variantes_all if _is_instock(v.get("availability"))]
    if not variantes_instock:
        continue

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
        "color": produto.get("color", ""),
        "age_group": _norm_age_group(produto.get("age_group", "")),
        "modalidade": produto.get("modalidade", ""),
        "signals": {
            "stock_ratio": round(mpn_stock_ratio.get(mpn, 0.0), 4),
            "novidade": round(mpn_novidade_score.get(mpn, 0.0), 4)
        },
        "variantes": variantes_all,
        "sugestoes": sugestoes_dict.get(id_base, sugestoes_dict.get(produto["id"], []))
    })

saida_json_final = {
    "gerado_em": datetime.utcnow().isoformat(),
    "pesos": {"sim": W_SIM, "stock": W_STOCK, "novidade": W_NEW},
    "produtos": saida_json
}

with open("upselling_final.json", "w", encoding="utf-8") as f:
    json.dump(saida_json_final, f, ensure_ascii=False, indent=2)

with open("produtos_sem_sugestoes.json", "w", encoding="utf-8") as f:
    json.dump(produtos_sem_sugestoes, f, ensure_ascii=False, indent=2)

print("✅ JSON final criado: upselling_final.json")
print("📁 Log criado: produtos_sem_sugestoes.json")

# =========================
# 📤 Upload para GitHub
# =========================
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
