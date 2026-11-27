import os
from pathlib import Path
import sqlite3
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, "servermind.db")
DB = os.path.abspath(DB)

if os.path.isdir(DB):
    raise SystemExit(
        f"Erro: o caminho informado é um diretório, não um arquivo SQLite: {DB}\n"
        "Próxima ação: informe o caminho completo de um arquivo .db, ex.: python .\\scripts\\abrir-sqlite.py \"C:\\caminho\\servermind.db\""
    )

if not os.path.isfile(DB):
    raise SystemExit(
        f"Banco não encontrado: {DB}\n"
        "Próxima ação: execute o ServerMind ao menos uma vez ou passe o caminho de um banco existente como argumento."
    )

uri = Path(DB).resolve().as_uri() + "?mode=ro"
try:
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
except sqlite3.Error as exc:
    raise SystemExit(
        f"Não foi possível abrir o banco SQLite em modo somente leitura: {DB}\n"
        f"Detalhe técnico: {exc}\n"
        "Próxima ação: confirme se o arquivo é um banco SQLite válido e se há permissão de leitura."
    )

print(f"SQLite em modo somente leitura: {DB}")
print("Tabelas:", ", ".join(tables) if tables else "(nenhuma)")
print("Digite SELECT ... para consultar, .tables para listar tabelas, .help para ajuda, .quit para sair.")

try:
    while True:
        try:
            sql = input("sqlite> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sql:
            continue
        cmd = sql.lower()
        if cmd in (".quit", ".exit", "quit", "exit"):
            break
        if cmd == ".tables":
            print("\n".join(tables) if tables else "(nenhuma)")
            continue
        if cmd == ".help":
            print("Consultas aceitas: SELECT ... ou WITH ... SELECT ... (somente leitura).")
            print("Exemplo: SELECT id, title, model, created_at FROM conversations ORDER BY id DESC LIMIT 20;")
            continue
        first_token = cmd.split(None, 1)[0]
        if first_token not in ("select", "with"):
            print("Bloqueado: apenas SELECT/WITH são permitidos; o banco está em modo somente leitura.")
            continue
        try:
            rows = conn.execute(sql).fetchall()
            if not rows:
                print("(nenhum registro)")
                continue
            cols = rows[0].keys()
            widths = {c: max(len(c), *(len(str(row[c]) if row[c] is not None else "NULL") for row in rows)) for c in cols}
            print(" | ".join(c.ljust(widths[c]) for c in cols))
            print("-+-".join("-" * widths[c] for c in cols))
            for row in rows:
                print(" | ".join((str(row[c]) if row[c] is not None else "NULL").ljust(widths[c]) for c in cols))
        except sqlite3.Error as exc:
            print(f"Erro SQL: {exc}")
finally:
    conn.close()
