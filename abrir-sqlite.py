import os
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "servermind.db")
DB = os.path.abspath(DB)
if not os.path.isfile(DB):
    raise SystemExit(f"Banco não encontrado: {DB}")

uri = "file:" + DB.replace("\\", "/") + "?mode=ro"
conn = sqlite3.connect(uri, uri=True)
conn.row_factory = sqlite3.Row
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
print(f"SQLite em modo somente leitura: {DB}")
print("Tabelas:", ", ".join(tables) if tables else "(nenhuma)")
print("Digite SELECT ... para consultar, .tables para listar tabelas, .help para ajuda, .quit para sair.")

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
        print("Exemplo: SELECT id, title, model FROM conversations ORDER BY updated_at DESC LIMIT 20;")
        continue
    if not (cmd.startswith("select ") or cmd.startswith("with ")):
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

conn.close()
