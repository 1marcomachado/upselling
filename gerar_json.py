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

# =========================
# 🔧 Helpers
# =========================
def _get_text(entry, tag, ns):
    el = entry.find(tag, ns)
    return el.text.strip() if el is not None and el.text else ""

def _split_cat(cat):
    return [c.strip() for c in cat.split(">")] if cat else []

def _first_level(cat):
    parts = _split_cat(cat)
    return parts[0].casefold() if parts else ""

def _is_acessorios_lista(categorias_validas):
    # Exceção ativa se QUALQUER categoria válida começar por "Acessórios"
    if not categorias_validas:
        return False
    for c in categorias_validas:
        if _first_level(c) == "acessórios":
            return True
    return False

def _is_instock(av):
    return ((av or "").strip().lower() == "in stock")

def _build_title_obj(produto):
    """
    Constrói o objeto de títulos multi-idioma com fallbacks:
    1) usa title_pt/es/en se existirem
    2) senão tenta 'title' legacy
    3) último recurso: mpn
    """
    legacy = produto.get("title", "")
    pt = (produto.get("title_pt") or "").strip() or legacy
    es = (produto.get("title_es") or "").strip() or legacy or pt
    en = (produto.get("title_en") or "").strip() or legacy or pt
    if not any([pt, es, en]):
        fallback = produto.get("mpn", "")
        pt = es = en = fallback
    return {"pt": pt, "es": es, "en": en}

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
    id_ = _get_text(entry, 'g:id', ns)
    mpn = _get_text(entry, 'g:mpn', ns)

    # ---- títulos multi-idioma + legacy ----
    title_pt = _get_text(entry, 'g:title_pt', ns)
    title_es = _get_text(entry, 'g:title_es', ns)
    title_en = _get_text(entry, 'g:title_en', ns)
    title_legacy = _get_text(entry, 'g:title', ns)  # opcional (fallback)

    category = _get_text(entry, 'g:category', ns)
    brand = _get_text(entry, 'g:brand', ns)
    image_link = _get_text(entry, 'g:image_link', ns)
    site = _get_text(entry, 'g:site', ns)
    gender = _get_text(entry, 'g:gender', ns)
    pack = _get_text(entry, 'g:pack', ns)
    price = _get_text(entry, 'g:price', ns)
    sale_price = _get_text(entry, 'g:sale_price', ns)
    size = _get_text(entry, 'g:size', ns)
    availability = _get_text(entry, 'g:availability', ns)

    produtos.append({
        "id": id_,
        "mpn": mpn,
        "title_pt": title_pt,
        "title_es": title_es,
        "title_en": title_en,
        "title": title_legacy,  # legacy para fallback
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
            response = session.get(img_url, headers=headers, timeout=10)
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

# 🧠 Preparar embeddings com cache (versão original com cache)
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

# 🔄 Construir sugestões (com exceção de Acessórios a ignorar brand/gender,
#     MAS no site == "2" a exceção NÃO se aplica: respeita brand/gender)
sugestoes_dict = {}
produtos_sem_sugestoes = []

for p in produtos_validos:
    base_mpn = p["mpn"]
    if base_mpn not in mpn_to_embedding:
        continue

    i = mpn_list.index(base_mpn)

    # 1) candidatos brutos: mesmo site
    candidatos_indices = []
    for j, mpn_cand in enumerate(mpn_list):
        if mpn_cand == base_mpn:
            continue
        cand = mpn_to_produto[mpn_cand]
        if (cand.get("site") or "") != (p.get("site") or ""):
            continue
        candidatos_indices.append(j)

    # 2) regras por categoria (OPÇÃO A: usar CAMINHO COMPLETO)
    categorias_validas = None
    acessorios_excecao = False

    cat_full = (p["category"] or "").strip()
    if cat_full in categorias_complementares:
        categorias_validas = categorias_complementares[cat_full]
        acessorios_excecao = _is_acessorios_lista(categorias_validas)

    # 🔵 No site "2", mesmo que seja "Acessórios", NÃO aplicar a exceção; respeitar brand/gender
    if acessorios_excecao and (str(p.get("site", "")).strip() != "2"):
        # Exceção: ignorar brand e gender; manter apenas categorias válidas (mesmo site já aplicado)
        candidatos_indices = [
            j for j in candidatos_indices
            if (mpn_to_produto[mpn_list[j]]["category"] or "").strip() in categorias_validas
        ]
    else:
        # Regra original: mesmo brand e gender + (se houver) categorias válidas
        candidatos_indices = [
            j for j in candidatos_indices
            if (mpn_to_produto[mpn_list[j]]["brand"] == p["brand"]) and
               (mpn_to_produto[mpn_list[j]]["gender"] == p["gender"]) and
               (
                   (categorias_validas is None) or
                   ((mpn_to_produto[mpn_list[j]]["category"] or "").strip() in categorias_validas)
               )
        ]

    # 3) ranking por similaridade (limite 30 + diversidade 20%)
    if candidatos_indices:
        diversity_quota = max(1, int(TOTAL_LIMIT * DIVERSITY_RATIO))
        max_per_cat = max(2, int(TOTAL_LIMIT * SOFT_CAP_RATIO))

        # Mapa categoria -> [(indice, score)]
        cat_to_candidates = defaultdict(list)
        for j in candidatos_indices:
            cand = mpn_to_produto[mpn_list[j]]
            cat = (cand["category"] or "").strip()
            score = similarity_matrix[i][j]
            cat_to_candidates[cat].append((j, score))

        # Ordenar candidatos dentro de cada categoria por score desc
        for cat in cat_to_candidates:
            cat_to_candidates[cat].sort(key=lambda x: x[1], reverse=True)

        # Quais categorias queremos cobrir?
        cats_presentes = list(cat_to_candidates.keys())
        if categorias_validas:
            # manter a ordem de categorias válidas, mas só as que têm candidatos
            cats_presentes = [c for c in categorias_validas if c in cat_to_candidates]
            # Se não houver nenhuma das válidas, fica o que existir
            if not cats_presentes:
                cats_presentes = list(cat_to_candidates.keys())
        else:
            # ordenar categorias pelo melhor score
            cats_presentes.sort(key=lambda c: cat_to_candidates[c][0][1], reverse=True)

        # Diversidade mínima: 1 por categoria até ao diversity_quota (priorizar categorias com melhor top-score)
        cats_ordenadas_por_topscore = sorted(
            cats_presentes,
            key=lambda c: cat_to_candidates[c][0][1],
            reverse=True
        )

        sugestoes_indices = []
        cat_counts = defaultdict(int)
        usados = set()

        for cat in cats_ordenadas_por_topscore[:diversity_quota]:
            idx_top, _ = cat_to_candidates[cat][0]
            if idx_top not in usados:
                sugestoes_indices.append(idx_top)
                usados.add(idx_top)
                cat_counts[cat] += 1

        # Pool global dos restantes candidatos, ordenados por score desc
        pool_global = sorted(
            [j for j in candidatos_indices if j not in usados],
            key=lambda x: similarity_matrix[i][x],
            reverse=True
        )

        # Preencher por relevância, respeitando teto por categoria
        for j in pool_global:
            if len(sugestoes_indices) >= TOTAL_LIMIT:
                break
            cat = (mpn_to_produto[mpn_list[j]]["category"] or "").strip()
            if cat_counts[cat] < max_per_cat:
                sugestoes_indices.append(j)
                cat_counts[cat] += 1

        # Se ainda faltarem lugares (p.ex., tetos muito restritivos), relaxar o teto
        if len(sugestoes_indices) < TOTAL_LIMIT:
            restantes_relax = [j for j in pool_global if j not in set(sugestoes_indices)]
            for j in restantes_relax:
                if len(sugestoes_indices) >= TOTAL_LIMIT:
                    break
                sugestoes_indices.append(j)

        # Converter indices -> mpn e guardar
        sugestoes = [mpn_list[idx] for idx in sugestoes_indices[:TOTAL_LIMIT]]
        # Guardar por ID base do produto (usar produto atual p)
        sugestoes_dict[p["id"]] = sugestoes

    else:
        produtos_sem_sugestoes.append({
            "id": p["id"],
            "mpn": p["mpn"],
            "title": _build_title_obj(p),  # multi-idioma
            "category": p["category"],
            "brand": p["brand"],
            "site": p["site"],
            "gender": p["gender"],
            "pack": p["pack"],
            "motivo": "Sem sugestões válidas (exceção Acessórios ativa)" if (acessorios_excecao and str(p.get("site","")).strip() != "2")
                      else "Sem sugestões visuais válidas"
        })

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
        "title": _build_title_obj(produto),  # objeto {pt, es, en}
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
get_resp = session.get(api_url, headers=headers, timeout=30)
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
put_resp = session.put(api_url, headers=headers, json=payload, timeout=30)
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
log_get_resp = session.get(log_api_url, headers=headers, timeout=30)
log_sha = log_get_resp.json().get("sha") if log_get_resp.status_code == 200 else None
log_payload = {
    "message": "Atualizar log de produtos sem sugestões",
    "content": log_content,
    "branch": branch
}
if log_sha:
    log_payload["sha"] = log_sha
log_put_resp = session.put(log_api_url, headers=headers, json=log_payload, timeout=30)
if log_put_resp.status_code in [200, 201]:
    print("✅ Log de produtos sem sugestões enviado para o GitHub.")
else:
    try:
        print("❌ Erro ao enviar log para o GitHub:", log_put_resp.json())
    except Exception:
        print("❌ Erro ao enviar log para o GitHub (sem JSON)")
