#!/usr/bin/env python3
"""Deterministic health check for the project wiki (no LLM, no dependencies).

Usage:
  python3 scripts/wiki_lint.py                 # report; exit 1 if there are errors
  python3 scripts/wiki_lint.py --write-index   # also regenerate wiki/index.md from page summaries

Rules come from wiki/SCHEMA.md. Checks:
  frontmatter   required fields and allowed type/status values per page type
  links         every relative markdown link resolves (file, directory, and #anchor); no [[wikilinks]]
  numbers       a number followed by [`dotted.key`](path/to/file.json) must match that JSON value,
                rounded as displayed (the "state rule": pages quote data, the data lives in the run)
  evidence      every finding cites at least one record, run, raw source or config, not only wiki pages
  budgets       page length per type
  orphans       wiki pages with no inbound link from another wiki page (index/log/hot don't count)
  registry      every experiment record and every run directory is linked from wiki/experiments.md,
                and every run listed in a record's frontmatter exists
  index         wiki/index.md lists every page (regenerate with --write-index)
  log           entries use `## [YYYY-MM-DD] kind | title`
  open          number of `Status: Unresolved` items (reported, not an error)
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
RECORDS = ROOT / "experiments"
RUNS = ROOT / "results" / "runs"

TYPES = {
    "project": ({"active", "archived"}, 160),
    "rq": ({"active"}, 80),
    "finding": ({"tentative", "supported", "contested", "superseded", "refuted"}, 40),
    "system": ({"active", "planned", "retired"}, 80),
    "concept": ({"active"}, 60),
    "paper": ({"unread", "read", "ingested"}, 60),
    "method": ({"active", "superseded"}, None),
}
SPECIAL_BUDGET = {"hot.md": 40}
NOT_INBOUND = {"index.md", "log.md", "hot.md"}
LOG_KINDS = {"experiment", "ingest", "decision", "finding", "lint", "schema"}
EVIDENCE_ROOTS = ("experiments", "results", "raw", "proposal", "configs")

LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
NUMBER_CITE = re.compile(r"(\d+(?:\.\d+)?)[^\d\[]{0,40}?\(\[`([^`]+)`\]\(([^)\s]+\.json)\)\)")


def frontmatter(text):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 4)
    if end < 0:
        return None, text
    fm = {}
    for line in text[4:end].splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip() for v in val[1:-1].split(",") if v.strip()]
            fm[key] = val
    return fm, text[end + 4:]


def slug(heading):
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(path, cache={}):
    if path not in cache:
        seen, out = {}, set()
        in_code = False
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("```"):
                in_code = not in_code
            m = re.match(r"^#{1,6}\s+(.*)$", line)
            if m and not in_code:
                base = slug(m.group(1))
                n = seen.get(base, 0)
                out.add(base if n == 0 else f"{base}-{n}")
                seen[base] = n + 1
        cache[path] = out
    return cache[path]


def json_value(path, key):
    d = json.loads(path.read_text())
    for part in key.split("."):
        d = d[part]
    return d


def md_files(base):
    return sorted(p for p in base.rglob("*.md") if ".obsidian" not in p.parts)


def strip_code(text):
    """Drop fenced code blocks and inline code spans so example links in them aren't checked."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-index", action="store_true")
    args = ap.parse_args()

    errors, warnings, info = [], [], []
    pages = md_files(WIKI)
    records = md_files(RECORDS)
    inbound = {p: 0 for p in pages}
    meta = {}

    if args.write_index:
        write_index(pages)
        pages = md_files(WIKI)
        inbound = {p: 0 for p in pages}

    for p in pages + records:
        rel = p.relative_to(ROOT)
        text = p.read_text(errors="replace")
        fm, body = frontmatter(text)
        meta[p] = fm or {}
        is_wiki = WIKI in p.parents

        # frontmatter
        if fm is None:
            errors.append(f"{rel}: missing frontmatter")
        elif is_wiki:
            for k in ("type", "summary", "status", "updated"):
                if k not in fm:
                    errors.append(f"{rel}: frontmatter missing `{k}`")
            t = fm.get("type")
            if t not in TYPES:
                errors.append(f"{rel}: unknown type `{t}`")
            elif fm.get("status") not in TYPES[t][0]:
                errors.append(f"{rel}: status `{fm.get('status')}` not allowed for type {t}")
            if t == "finding":
                for k in ("confidence", "rq", "sources"):
                    if k not in fm:
                        errors.append(f"{rel}: finding missing `{k}`")
                if fm.get("confidence") not in (None, "high", "medium", "low"):
                    errors.append(f"{rel}: confidence must be high|medium|low")
        elif fm.get("type") != "experiment-record":
            errors.append(f"{rel}: experiment record needs `type: experiment-record`")

        # budgets
        if is_wiki and fm:
            n = text.count("\n") + 1
            budget = SPECIAL_BUDGET.get(p.name) or TYPES.get(fm.get("type"), (None, None))[1]
            if budget and n > budget and p.name not in ("index.md",):
                warnings.append(f"{rel}: {n} lines, over the {budget}-line budget for its type")

        # links and anchors
        plain = strip_code(body)
        if "[[" in plain:
            errors.append(f"{rel}: uses [[wikilinks]]; use relative markdown links")
        cites_evidence = False
        for _, target in LINK.findall(plain):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            dest = (p.parent / path_part).resolve() if path_part else p
            if not dest.exists():
                errors.append(f"{rel}: broken link -> {target}")
                continue
            if anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                errors.append(f"{rel}: missing anchor #{anchor} in {dest.relative_to(ROOT)}")
            if dest in inbound and dest != p and p.name not in NOT_INBOUND:
                inbound[dest] += 1
            try:
                top = dest.relative_to(ROOT).parts[0]
            except ValueError:
                top = ""
            if top in EVIDENCE_ROOTS:
                cites_evidence = True
        if is_wiki and fm and fm.get("type") == "finding" and not cites_evidence:
            errors.append(f"{rel}: finding cites no record, run, raw source or config")

        # numbers quoted from JSON (the citation may sit on the next line)
        for num, key, target in NUMBER_CITE.findall(body):
            dest = (p.parent / target).resolve()
            if not dest.exists():
                continue  # reported as a broken link above
            try:
                val = json_value(dest, key)
            except (KeyError, TypeError, json.JSONDecodeError):
                errors.append(f"{rel}: key `{key}` not found in {dest.relative_to(ROOT)}")
                continue
            decimals = len(num.split(".")[1]) if "." in num else 0
            if round(float(val), decimals) != float(num):
                errors.append(f"{rel}: quotes {num} for `{key}`, file has {val}")

    # orphans
    for p, n in inbound.items():
        if n == 0 and p.name not in NOT_INBOUND | {"SCHEMA.md", "open-questions.md", "experiments.md"}:
            warnings.append(f"{p.relative_to(ROOT)}: orphan (no inbound link from another wiki page)")

    # registry
    registry = WIKI / "experiments.md"
    reg_text = registry.read_text() if registry.exists() else ""
    for r in records:
        if f"../experiments/{r.name}" not in reg_text:
            errors.append(f"registry: {r.relative_to(ROOT)} not linked from wiki/experiments.md")
        for run in meta.get(r, {}).get("runs", []) or []:
            if not (RUNS / run).is_dir():
                errors.append(f"{r.relative_to(ROOT)}: frontmatter lists missing run {run}")
    for d in sorted(x for x in RUNS.iterdir() if x.is_dir()) if RUNS.exists() else []:
        if f"results/runs/{d.name}/" not in reg_text:
            errors.append(f"registry: run directory {d.name} not linked from wiki/experiments.md")

    # index
    index = WIKI / "index.md"
    idx_text = index.read_text() if index.exists() else ""
    for p in pages:
        if p.name != "index.md" and f"]({p.relative_to(WIKI).as_posix()})" not in idx_text:
            errors.append(f"index: {p.relative_to(ROOT)} not listed (run with --write-index)")

    # log format
    log = WIKI / "log.md"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith("## ") and not re.match(r"^## \[\d{4}-\d{2}-\d{2}\] (\w+) \| .+", line):
                errors.append(f"log: bad entry heading: {line}")
            m = re.match(r"^## \[\d{4}-\d{2}-\d{2}\] (\w+) \|", line)
            if m and m.group(1) not in LOG_KINDS:
                errors.append(f"log: unknown kind `{m.group(1)}`: {line}")

    unresolved = sum(len(re.findall(r"^- Status: Unresolved", p.read_text(), re.M)) for p in pages)
    info.append(f"{len(pages)} wiki pages, {len(records)} records, {unresolved} unresolved open question(s)")

    for label, items in (("ERROR", errors), ("WARN", warnings)):
        for it in items:
            print(f"{label}  {it}")
    for it in info:
        print(f"INFO   {it}")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


SECTIONS = [
    ("Start here", ["hot.md", "SCHEMA.md", "project/overview.md", "project/milestones.md", "experiments.md",
                    "open-questions.md", "log.md"]),
    ("Project", "project/"),
    ("Research questions", "rq/"),
    ("Findings", "findings/"),
    ("Systems (engines, devices, models, cluster)", "systems/"),
    ("Concepts", "concepts/"),
    ("Papers", "papers/"),
    ("Methods", "methods/"),
]


def write_index(pages):
    listed, lines = set(), []
    lines += ["---", "type: project", "summary: Catalog of every wiki page with a one-line summary (generated by scripts/wiki_lint.py --write-index).",
              "status: active", "updated: generated", "---", "", "# Wiki index", "",
              "Generated from each page's `summary` field; don't edit by hand. Rules: [SCHEMA.md](SCHEMA.md).", ""]
    for title, sel in SECTIONS:
        if isinstance(sel, list):
            chosen = [WIKI / s for s in sel if (WIKI / s).exists()]
        else:
            chosen = [p for p in pages if p.relative_to(WIKI).as_posix().startswith(sel)]
        chosen = [p for p in chosen if p not in listed and p.name != "index.md"]
        if not chosen:
            continue
        lines += [f"## {title}", ""]
        for p in chosen:
            fm, body = frontmatter(p.read_text(errors="replace"))
            summary = (fm or {}).get("summary", "(no summary)")
            status = (fm or {}).get("status", "")
            tag = f" `{status}`" if status and status not in ("active",) else ""
            h1 = re.search(r"^# (.+)$", body, re.M)
            title = h1.group(1).strip() if h1 else p.stem
            lines.append(f"- **[{title}]({p.relative_to(WIKI).as_posix()})**{tag}: {summary}")
            listed.add(p)
        lines.append("")
    (WIKI / "index.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
