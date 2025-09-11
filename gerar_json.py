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

# ===== NOVOS IMPORTS (textura) =====
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2

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

# ---- divisão da similaridade entre CNN e textura
W_SIM_CNN  = 0.70 * W_SIM
W_SIM_TEX  = 0.30 * W_SIM

NOVELTY_HALF_LIFE_DAYS = 180
DEFAULT_NEWNESS_WHEN_MISSING = 0.0

# ===== parâmetros de textura =====
LBP_N_POINTS = 8
LBP_RADIUS   = 1
LBP_METHOD   = "uniform"  # ~59 bins
GLCM_DIST    = [1, 2, 4]
GLCM_ANG     = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Flag para ligar/desligar textura via ENV (ex.: TEXTURE_ENABLED=0)
TEXTURE_ON = os.getenv("TEXTURE_ENABLED", "1") == "1"

# =========================
# 🔐 Env / Endpoints
# =========================
load_dotenv()  # útil em local

# Em Actions, já existem GITHUB_TOKEN e GITHUB_REPOSITORY
token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
repo  = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")
branch = os.getenv("GITHUB_REF_NAME", "main")
if not repo:
    raise RuntimeError("Repo não definido (esperava GITHUB_REPOSITORY ou GITHUB_REPO).")

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
r = session.get(xml_url, timeout=60)
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

# ===== Helpers imagem / textura =====
def load_pil_image(img_url, image_path):
    try:
        if os.path.exists(image_path):
            pil = Image.open(image_path).convert('RGB')
        else:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(img_url, headers=headers, timeout=20)
            if "image" not in response.headers.get("Content-Type", ""):
                print(f"❌ Ignorado (não é imagem): {img_url}")
                return None
            pil = Image.open(BytesIO(response.content)).convert('RGB')
            pil.save(image_path)
        # pequena validação
        if min(pil.size) < 80:
            print(f"⚠️ Imagem muito pequena, pode prejudicar textura: {image_path}")
        return pil
    except Exception as e:
        print(f"⚠️ Erro ao carregar imagem {img_url}: {e}")
        return None

def get_cnn_embedding_from_pil(pil):
    try:
        img = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"⚠️ Erro CNN: {e}")
        return None

def _prep_gray_for_texture(pil_img: Image.Image) -> np.ndarray:
    # crop central (reduz fundo e bolsas/brancos)
    w, h = pil_img.size
    c = int(min(w, h) * 0.8)
    left = (w - c)//2; top = (h - c)//2
    pil_c = pil_img.crop((left, top, left+c, top+c))

    # PIL RGB -> OpenCV BGR -> gray + CLAHE
    img = np.array(pil_c)[:, :, ::-1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def _glcm_feats(gray: np.ndarray) -> np.ndarray:
    q = (gray / 8).astype(np.uint8)  # 0..31
    glcm = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANG, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    vals = [graycoprops(glcm, p).mean() for p in props]
    return np.array(vals, dtype=np.float32)

def extract_texture_vector(pil_img: Image.Image) -> np.ndarray:
    g = _prep_gray_for_texture(pil_img)
    v = np.concatenate([_lbp_hist(g), _glcm_feats(g)], axis=0)  # ~65 dims
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)

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

# ===== cache de TEXTURA =====
tex_file = "texture_vectors.npy"
tex_mpns_file = "mpns_texture.json"

if os.path.exists(tex_file) and os.path.exists(tex_mpns_file):
    texture_vectors = np.load(tex_file)
    with open(tex_mpns_file, "r", encoding="utf-8") as f:
        mpns_tex_existentes = json.load(f)
else:
    texture_vectors = []
    mpns_tex_existentes = []

mpn_to_texture = {}
for i, mpn in enumerate(mpns_tex_existentes):
    mpn_to_texture[mpn] = texture_vectors[i]

# ===== construir/atualizar caches =====
novos_embeddings = []
novas_texturas = []
mpns_adicionados = set(mpns_existentes)
mpns_tex_adicionados = set(mpns_tex_existentes)

for p in produtos:
    mpn = p["mpn"]
    if not mpn:
        continue

    image_filename = f"{mpn}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    pil = load_pil_image(p["image_link"], image_path)
    if pil is None:
        continue

    # CNN
    if mpn not in mpns_adicionados:
        emb = get_cnn_embedding_from_pil(pil)
        if emb is not None:
            novos_embeddings.append(emb)
            mpn_to_embedding[mpn] = emb
            mpn_to_produto[mpn] = p
            mpns_adicionados.add(mpn)

    # TEXTURA
    if TEXTURE_ON and mpn not in mpns_tex_adicionados:
        try:
            tex = extract_texture_vector(pil)
            novas_texturas.append(tex)
            mpn_to_texture[mpn] = tex
            mpns_tex_adicionados.add(mpn)
        except Exception as e:
            print(f"⚠️ Erro textura para {mpn}: {e}")

# guardar caches
if novos_embeddings:
    novos_embeddings = np.array(novos_embeddings)
    embeddings = np.vstack((embeddings, novos_embeddings)) if len(embeddings) else novos_embeddings
    np.save(embeddings_file, embeddings)
    with open(mpns_file, "w", encoding="utf-8") as f:
        json.dump(list(mpns_adicionados), f, ensure_ascii=False, indent=2)

if TEXTURE_ON and novas_texturas:
    novas_texturas = np.array(novas_texturas)
    texture_vectors = np.vstack((texture_vectors, novas_texturas)) if len(texture_vectors) else novas_texturas
    np.save(tex_file, texture_vectors)
    with open(tex_mpns_file, "w", encoding="utf-8") as f:
        json.dump(list(mpns_tex_adicionados), f, ensure_ascii=False, indent=2)

# filtrar produtos com embeddings CNN
produtos_validos = []
for p in produtos:
    mpn = p["mpn"]
    if mpn in mpn_to_embedding:
        produtos_validos.append(p)
        mpn_to_produto[mpn] = p

print(f"✅ Produtos com embeddings CNN: {len(produtos_validos)}")
if TEXTURE_ON:
    print(f"✅ Vetores de textura disponíveis para: {len(mpn_to_texture)} MPNs")

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
# 📊 Similaridades (CNN + TEX)
# =========================
print("📊 A calcular similaridades...")
mpn_list = list(mpn_to_embedding.keys())

# CNN
mpn_embeddings = np.array([mpn_to_embedding[m] for m in mpn_list])
# L2 norm defensiva
mpn_embeddings = mpn_embeddings / (np.linalg.norm(mpn_embeddings, axis=1, keepdims=True) + 1e-8)
similarity_matrix_cnn = cosine_similarity(mpn_embeddings)

# TEX
if TEXTURE_ON and len(mpn_to_texture) > 0:
    tex_dim = next(iter(mpn_to_texture.values())).shape[0]
    tex_matrix = np.zeros((len(mpn_list), tex_dim), dtype=np.float32)
    for idx, m in enumerate(mpn_list):
        tex_matrix[idx] = mpn_to_texture.get(m, np.zeros(tex_dim, dtype=np.float32))
    tex_matrix = tex_matrix / (np.linalg.norm(tex_matrix, axis=1, keepdims=True) + 1e-8)
    similarity_matrix_tex = cosine_similarity(tex_matrix)
else:
    similarity_matrix_tex = None

def _combined_score(i_idx: int, j_idx: int) -> float:
    sim_cnn = float(similarity_matrix_cnn[i_idx][j_idx])
    sim_tex = 0.0
    if TEXTURE_ON and similarity_matrix_tex is not None:
        sim_tex = float(similarity_matrix_tex[i_idx][j_idx])

    cand_mpn = mpn_list[j_idx]
    stock = mpn_stock_ratio.get(cand_mpn, 0.0)
    newn  = mpn_novidade_score.get(cand_mpn, DEFAULT_NEWNESS_WHEN_MISSING)

    sim_part = W_SIM_CNN * _clamp01(sim_cnn) + W_SIM_TEX * _clamp01(sim_tex)
    return float(sim_part + W_STOCK * _clamp01(stock) + W_NEW * _clamp01(newn))

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
# 📝 JSON final (igual ao último, com "texture" opcional)
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

    # sinal opcional de energia de textura (norma do vetor)
    if TEXTURE_ON and (mpn in mpn_to_texture):
        tex_energy = float(np.linalg.norm(mpn_to_texture.get(mpn)))
    else:
        tex_energy = 0.0

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
            "novidade": round(mpn_novidade_score.get(mpn, 0.0), 4),
            "texture": round(tex_energy, 4)
        },
        "variantes": variantes_all,
        "sugestoes": sugestoes_dict.get(id_base, sugestoes_dict.get(produto["id"], []))
    })

saida_json_final = {
    "gerado_em": datetime.utcnow().isoformat(),
    "pesos": {
        "sim_total": W_SIM,
        "sim_cnn": W_SIM_CNN,
        "sim_tex": W_SIM_TEX if TEXTURE_ON else 0.0,
        "stock": W_STOCK,
        "novidade": W_NEW
    },
    "textura_ativa": TEXTURE_ON,
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
    # Em Actions, usa "Bearer"; em local também funciona
    "Authorization": f"Bearer {token}" if token else "",
    "Accept": "application/vnd.github.v3+json"
}

# JSON principal
get_resp = requests.get(api_url, headers=headers, timeout=30)
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
put_resp = requests.put(api_url, headers=headers, json=payload, timeout=60)
if put_resp.status_code in [200, 201]:
    print("✅ JSON copiado para o GitHub com sucesso.")
else:
    try:
        print("❌ Erro ao enviar para o GitHub:", put_resp.status_code, put_resp.json())
    except Exception:
        print("❌ Erro ao enviar para o GitHub (sem JSON)")

# Log
log_filename = "produtos_sem_sugestoes.json"
log_api_url = f"https://api.github.com/repos/{repo}/contents/{log_filename}"
with open(log_filename, "rb") as f:
    log_content = base64.b64encode(f.read()).decode()
log_get_resp = requests.get(log_api_url, headers=headers, timeout=30)
log_sha = log_get_resp.json().get("sha") if log_get_resp.status_code == 200 else None
log_payload = {
    "message": "Atualizar log de produtos sem sugestões",
    "content": log_content,
    "branch": branch
}
if log_sha:
    log_payload["sha"] = log_sha
log_put_resp = requests.put(log_api_url, headers=headers, json=log_payload, timeout=60)
if log_put_resp.status_code in [200, 201]:
    print("✅ Log de produtos sem sugestões enviado para o GitHub.")
else:
    try:
        print("❌ Erro ao enviar log para o GitHub:", log_put_resp.status_code, log_put_resp.json())
    except Exception:
        print("❌ Erro ao enviar log para o GitHub (sem JSON)")
