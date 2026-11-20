#!/usr/bin/env python3
"""Build and query a SQLite FTS5 index of CLI tool descriptions and options."""

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Curated high-signal CLI schemas (guaranteed clean even on minimal systems)
CURATED_SCHEMAS: List[Tuple[str, str, str, str]] = [
    ("find", "search files directories recursively by name size mtime type permissions",
     "-name -iname -type (f|d|l) -size (+10M|-1k) -mtime (+7|-1) -maxdepth -mindepth -exec {} + -delete -print0",
     "find /var/log -type f -name '*.log' -mtime +7 -delete"),
    ("grep", "search text patterns regex inside files recursively count invert match",
     "-r -R -i -n -l -v -E -F -P -w -c -H --include --exclude-dir -A -B -C",
     "grep -rnE --include='*.py' 'TODO|FIXME' src/"),
    ("rg", "ripgrep fast recursive regex text search files",
     "-i -n -l -v -F -w -c -g --glob -t --type -A -B -C --hidden",
     "rg -n -t py 'def train' ."),
    ("sed", "stream editor replace substitute delete lines in-place regex",
     "-i -E -n 's/old/new/g' '/pat/d' '1,10p'",
     "sed -i.bak -E 's/foo/bar/g' file.txt"),
    ("awk", "pattern scanning column field processing sum average text tabular",
     "-F ',' '{print $1, $NF}' NR FNR BEGIN END",
     "awk -F: '$3 >= 1000 {print $1}' /etc/passwd"),
    ("tar", "archive compress extract tarball gzip bzip2 xz zstd",
     "-c -x -t -v -f -z (gzip) -j (bzip2) -J (xz) --zstd -C (dir) --exclude",
     "tar -czvf backup.tar.gz --exclude='.git' project/"),
    ("rsync", "sync copy files directories incrementally remote ssh progress delete",
     "-a -v -z -h -P --progress --delete --dry-run -n --exclude -e ssh",
     "rsync -avzP --delete src/ user@host:/dst/"),
    ("xargs", "build execute command lines from standard input null-delimited parallel",
     "-0 -n -I {} -P (procs) -r --no-run-if-empty",
     "find . -name '*.tmp' -print0 | xargs -0 -r rm -f"),
    ("chmod", "change file mode permissions executable read write octal recursive",
     "-R +x -x u+rwx g-w o-rwx 644 755 600",
     "chmod -R 755 scripts/"),
    ("chown", "change file owner group recursive",
     "-R user:group --reference",
     "chown -R www-data:www-data /var/www/html"),
    ("ps", "list running processes pid cpu memory tree",
     "aux -ef -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem",
     "ps aux --sort=-%mem | head -n 10"),
    ("kill", "terminate send signal process pid pkill killall",
     "-9 (SIGKILL) -15 (SIGTERM) -HUP -l",
     "kill -15 12345"),
    ("ss", "socket statistics listening tcp udp ports network connections",
     "-t -u -l -n -p -a state",
     "ss -tulnp"),
    ("curl", "transfer data url http https download api header post json",
     "-s -S -L -f -o -O -I -X (GET|POST|PUT|DELETE) -H -d --data-binary -u",
     "curl -fsSL -H 'Accept: application/json' https://example.com/api"),
    ("du", "disk usage directory file size summary human readable sort",
     "-h -s -a -d --max-depth=1 --apparent-size --exclude",
     "du -sh * | sort -hr | head -n 20"),
    ("df", "disk space filesystem free usage human readable inodes",
     "-h -T -i --total",
     "df -hT"),
    ("sort", "sort lines text numeric human reverse unique key column",
     "-n -h -r -u -k -t -V",
     "sort -t, -k2,2nr data.csv"),
    ("uniq", "filter count duplicate adjacent lines",
     "-c -d -u -i",
     "sort access.log | uniq -c | sort -nr"),
    ("cut", "extract columns fields characters bytes from lines",
     "-d (delim) -f (fields) -c (chars)",
     "cut -d',' -f1,3 data.csv"),
    ("tr", "translate squeeze delete characters lowercase uppercase",
     "-d -s -c '[:lower:]' '[:upper:]'",
     "tr '[:upper:]' '[:lower:]' < in.txt"),
    ("head", "output first lines bytes of files",
     "-n -c -q",
     "head -n 20 app.log"),
    ("tail", "output last lines follow log file live",
     "-n -f -F -c",
     "tail -n 100 -f /var/log/syslog"),
    ("wc", "count lines words characters bytes",
     "-l -w -c -m",
     "wc -l *.py"),
    ("ln", "create symbolic soft hard links",
     "-s -f -n -v",
     "ln -sfn /opt/app-v2 /opt/app"),
    ("cp", "copy files directories preserve recursive force",
     "-r -a -v -u -n -f -p",
     "cp -av src/ dst/"),
    ("mv", "move rename files directories",
     "-v -i -n -f -u",
     "mv -v old.txt new.txt"),
    ("rm", "remove delete files directories recursive force",
     "-r -f -i -v -d",
     "rm -rf build/"),
    ("mkdir", "create directories parent paths mode",
     "-p -m -v",
     "mkdir -p src/components/ui"),
    ("systemctl", "manage systemd services start stop restart status enable logs",
     "status start stop restart reload enable disable list-units --failed --user",
     "systemctl restart nginx.service"),
    ("journalctl", "query systemd journal logs unit boot time follow",
     "-u -b -f -n --since --until -p (err|warning) --no-pager",
     "journalctl -u docker.service --since '1 hour ago' --no-pager"),
    ("git", "version control commit branch diff log status rebase stash checkout",
     "status diff log --oneline -n branch checkout -b switch commit -m pull push rebase stash",
     "git log --oneline --graph -n 20"),
    ("docker", "containers images build run exec logs ps compose prune",
     "ps -a images run --rm -it -p -v exec -it logs -f build -t system prune",
     "docker ps -a --format 'table {{.Names}}\\t{{.Status}}'"),
    ("jq", "json processor query filter transform keys array",
     "-r -c -e --arg '.items[] | select(.active) | .name'",
     "jq -r '.users[].email' data.json"),
    ("ffmpeg", "convert transcode trim compress audio video media",
     "-i -c:v -c:a -crf -b:a -ss -t -vf -y",
     "ffmpeg -i input.mp4 -c:v libx264 -crf 23 -c:a aac output.mp4"),
    ("ssh", "secure shell remote login tunnel port forward key",
     "-i -p -L -R -D -N -f -o StrictHostKeyChecking=no",
     "ssh -i ~/.ssh/id_ed25519 -L 8080:localhost:80 user@host"),
    ("scp", "secure copy files over ssh",
     "-r -P -i -C",
     "scp -r -P 2222 ./dist user@host:/var/www/"),
    ("pacman", "arch linux package manager install update search remove clean cache",
     "-S -Syu -Ss -Qs -Qi -Rns -Scc",
     "pacman -Syu"),
    ("apt", "debian ubuntu package manager install update upgrade search remove",
     "update upgrade install -y remove purge autoremove search list --installed",
     "apt update && apt install -y ripgrep"),
    ("ip", "network interface ip address route link show",
     "addr show route show link show -br -c",
     "ip -br addr show"),
    ("lsof", "list open files ports sockets by process",
     "-i :PORT -p PID -u USER +D DIR -n -P",
     "lsof -i :8080 -n -P"),
    ("crontab", "schedule periodic cron jobs edit list",
     "-l -e -u",
     "crontab -l"),
    ("tee", "read stdin write stdout and files append",
     "-a",
     "make 2>&1 | tee -a build.log"),
    ("diff", "compare files directories line by line unified patch",
     "-u -r -w -q --color=auto",
     "diff -ru dir_a/ dir_b/"),
    ("zip", "create zip archive compress recursive exclude",
     "-r -q -9 -x",
     "zip -r archive.zip folder/ -x '*.git*'"),
    ("unzip", "extract list zip archive",
     "-d -l -o -q",
     "unzip -q archive.zip -d out/"),
    ("gzip", "compress decompress .gz files keep stdout",
     "-d -k -c -9 -r",
     "gzip -dk file.txt.gz"),
    ("uname", "print kernel system architecture os version",
     "-a -r -m -s",
     "uname -a"),
    ("free", "display system memory ram swap usage",
     "-h -m -g -s",
     "free -h"),
    ("ls", "list directory contents permissions hidden human sort time size",
     "-l -a -h -t -r -S -1 -R",
     "ls -laht"),
]


def harvest_local_tool_flags(tool: str) -> str:
    """Safely inspect `--help` on the local machine to extract real flags."""
    if shutil.which(tool) is None:
        return ""
    try:
        proc = subprocess.run(
            [tool, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1.5,
            check=False,
        )
        text = proc.stdout or ""
        # Extract short and long flags like -r, --recursive, --max-depth
        flags = re.findall(r"(?<!\w)(--[a-zA-Z0-9][a-zA-Z0-9_-]{1,24}|-[a-zA-Z0-9])(?![a-zA-Z0-9_-])", text)
        seen = []
        for f in flags:
            if f not in seen:
                seen.append(f)
            if len(seen) >= 24:
                break
        return " ".join(seen)
    except Exception:
        return ""


def build_schema_db(db_path: str, harvest_local: bool = True) -> int:
    """Build SQLite FTS5 CLI Schema-RAG database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE cli_tools (
            tool TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            flags TEXT NOT NULL,
            example TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE cli_tools_fts USING fts5(
            tool,
            description,
            flags,
            content='cli_tools',
            content_rowid='rowid',
            tokenize='porter unicode61'
        )
        """
    )

    count = 0
    for tool, desc, flags, example in CURATED_SCHEMAS:
        merged_flags = flags
        if harvest_local:
            local_flags = harvest_local_tool_flags(tool)
            if local_flags:
                # Merge unique flags while keeping curated flags first
                existing = set(re.findall(r"--?[a-zA-Z0-9_-]+", flags))
                extras = [f for f in local_flags.split() if f not in existing][:8]
                if extras:
                    merged_flags = f"{flags} {' '.join(extras)}"

        cur.execute(
            "INSERT OR REPLACE INTO cli_tools (tool, description, flags, example) VALUES (?, ?, ?, ?)",
            (tool, desc, merged_flags, example),
        )
        count += 1

    # Populate FTS index
    cur.execute(
        """
        INSERT INTO cli_tools_fts(rowid, tool, description, flags)
        SELECT rowid, tool, description, flags FROM cli_tools
        """
    )
    conn.commit()
    conn.close()
    return count


class SchemaRAGRetriever:
    """Sub-2ms local SQLite FTS5 retriever for compact CLI flag cards."""

    # Lightweight keyword -> tool boost map for common natural language intents
    INTENT_HINTS: Dict[str, List[str]] = {
        "disk": ["df", "du"],
        "space": ["df", "du"],
        "memory": ["free", "ps"],
        "ram": ["free", "ps"],
        "process": ["ps", "kill"],
        "port": ["ss", "lsof"],
        "socket": ["ss", "lsof"],
        "archive": ["tar", "zip", "unzip"],
        "compress": ["tar", "gzip", "zip"],
        "extract": ["tar", "unzip", "gzip"],
        "replace": ["sed", "awk", "tr"],
        "substitute": ["sed", "tr"],
        "column": ["awk", "cut"],
        "field": ["awk", "cut"],
        "json": ["jq", "curl"],
        "download": ["curl", "rsync"],
        "service": ["systemctl", "journalctl"],
        "log": ["journalctl", "tail", "grep"],
        "permission": ["chmod", "chown"],
        "owner": ["chown", "chmod"],
        "symlink": ["ln", "find"],
        "link": ["ln"],
        "count": ["wc", "uniq", "grep"],
        "duplicate": ["uniq", "sort"],
        "video": ["ffmpeg"],
        "audio": ["ffmpeg"],
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def close(self):
        self.conn.close()

    def retrieve(self, nl_query: str, top_k: int = 3, max_chars: int = 420) -> Tuple[str, List[str]]:
        """Return compact [SCHEMA] text block and list of matched tool names."""
        tokens = re.findall(r"[a-zA-Z0-9_]+", nl_query.lower())
        stop = {
            "the", "a", "an", "to", "in", "of", "for", "on", "with", "all",
            "and", "or", "from", "by", "that", "this", "is", "are", "how",
            "command", "bash", "shell", "linux", "using", "use", "get", "show",
            "print", "display", "current", "directory", "file", "files",
        }
        keywords = [t for t in tokens if len(t) >= 2 and t not in stop]

        cur = self.conn.cursor()
        results: List[Tuple[str, str, str]] = []
        seen = set()

        # 1. Direct tool name mention gets highest priority
        for tok in tokens:
            row = cur.execute(
                "SELECT tool, flags, example FROM cli_tools WHERE tool = ?", (tok,)
            ).fetchone()
            if row and row[0] not in seen:
                results.append(row)
                seen.add(row[0])

        # 2. Intent hint boost
        for tok in tokens:
            for hinted in self.INTENT_HINTS.get(tok, []):
                if hinted not in seen and len(results) < top_k:
                    row = cur.execute(
                        "SELECT tool, flags, example FROM cli_tools WHERE tool = ?", (hinted,)
                    ).fetchone()
                    if row:
                        results.append(row)
                        seen.add(row[0])

        # 3. FTS5 BM25 search
        if len(results) < top_k and keywords:
            fts_query = " OR ".join(f'"{k}"' for k in keywords[:10])
            try:
                rows = cur.execute(
                    """
                    SELECT c.tool, c.flags, c.example
                    FROM cli_tools_fts f
                    JOIN cli_tools c ON c.rowid = f.rowid
                    WHERE cli_tools_fts MATCH ?
                    ORDER BY bm25(cli_tools_fts)
                    LIMIT ?
                    """,
                    (fts_query, top_k * 2),
                ).fetchall()
                for row in rows:
                    if row[0] not in seen:
                        results.append(row)
                        seen.add(row[0])
                        if len(results) >= top_k:
                            break
            except sqlite3.OperationalError:
                pass

        results = results[:top_k]
        if not results:
            return "", []

        lines = []
        total_len = 0
        tools_out = []
        for tool, flags, _example in results:
            entry = f"{tool}: {flags}"
            if total_len + len(entry) + 1 > max_chars and lines:
                break
            lines.append(entry)
            tools_out.append(tool)
            total_len += len(entry) + 1

        return "\n".join(lines), tools_out


def main():
    parser = argparse.ArgumentParser(description="Build local SQLite FTS5 CLI Schema-RAG DB")
    parser.add_argument("--db", type=str, default="./data/cli_schema.sqlite")
    parser.add_argument("--no-harvest", action="store_true", help="Skip local --help flag harvesting")
    parser.add_argument("--test-query", type=str, default="find all log files older than 7 days and delete them")
    args = parser.parse_args()

    n = build_schema_db(args.db, harvest_local=not args.no_harvest)
    print(f"[Schema-RAG] Built {args.db} with {n} CLI tool schemas.")

    retriever = SchemaRAGRetriever(args.db)
    card, tools = retriever.retrieve(args.test_query)
    print(f"[Schema-RAG] Sample query: {args.test_query!r}")
    print(f"[Schema-RAG] Matched tools: {tools}")
    print(f"[Schema-RAG] Injected card:\n{card}")
    retriever.close()


if __name__ == "__main__":
    main()
