
import os
from git import Repo

def git_push(repo_path, commit_message="Atualização automática de sugestões"):
    repo = Repo(repo_path)
    repo.git.add('sugestoes_upselling.json')
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    origin.push()

if os.path.isdir(".git"):
    try:
        git_push(".", "🤖 Atualização diária das sugestões de upselling")
        print("✅ JSON atualizado e enviado para o GitHub com sucesso.")
    except Exception as e:
        print("❌ Erro ao enviar para GitHub:", e)
else:
    print("⚠️ Este diretório não parece ser um repositório Git.")
