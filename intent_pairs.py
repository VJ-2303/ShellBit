"""Small GNU text-utility challenge. Run allowlisted filters on isolated fixture files."""

from itertools import product
import os
import hashlib
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

# Each family has two independently specified answers on two discriminating inputs.
FAMILIES = {
    "ordering": (
        ("Sort the lines of {file} lexicographically.", "sort", ("10\n2\n", "11\n3\n")),
        ("Sort the lines of {file} numerically.", "sort -n", ("2\n10\n", "3\n11\n")),
        ("2\n10\n", "11\n3\n"),
    ),
    "frequency": (
        ("Print distinct lines of {file} in sorted order.", "sort -u", ("a\nb\n", "x\ny\n")),
        ("Print only lines occurring once in {file}, sorted.", "sort | uniq -u", ("b\n", "x\n")),
        ("a\nb\na\n", "y\nx\ny\n"),
    ),
    "counting": (
        ("Count the lines in {file}.", "wc -l", ("2\n", "1\n")),
        ("Count the words in {file}.", "wc -w", ("3\n", "4\n")),
        ("one two\nthree\n", "four five six seven\n"),
    ),
    "endpoints": (
        ("Print the first line of {file}.", "head -n 1", ("first\n", "z\n")),
        ("Print the last line of {file}.", "tail -n 1", ("last\n", "x\n")),
        ("first\nmiddle\nlast\n", "z\ny\nx\n"),
    ),
}

# Stages are reconstructed from this table. No model text is evaluated by Bash.
FILTERS = {("sort",), ("sort", "-n"), ("sort", "-u"), ("uniq",), ("uniq", "-u"),
           ("head", "-n", "1"), ("tail", "-n", "1"), ("head", "-1"), ("tail", "-1"), ("cat",), ("wc", "-w"), ("wc", "-l")}


def composed_examples(split):
    """Change one ordering decision while holding the remaining request fixed."""
    files = {"train": ("scores.txt", "raw values.txt", "measurements.txt"),
             "dev": ("readings.txt", "trial scores.txt")}[split]
    for numeric, reverse, unique, count in product((False, True), (False, True),
                                                    (False, True), (1, 2, 3)):
        # Reserve the numeric + reverse composition across all filenames and lengths.
        if (numeric and reverse) != (split == "dev"):
            continue
        flags = [flag for flag, active in (("-n", numeric), ("-r", reverse), ("-u", unique)) if active]
        stage = " ".join(["sort", *flags])
        family = f"prefix-{int(numeric)}{int(reverse)}{int(unique)}-{count}"
        order = "numerically" if numeric else "in lexical order"
        direction = "descending" if reverse else "ascending"
        for filename in files:
            for side, end in enumerate(("first", "last")):
                pipeline = f"{stage} {shlex.quote(filename)} | {('head', 'tail')[side]} -n {count}"
                qualifiers = " and remove duplicate lines" if unique else " and keep duplicate lines"
                yield dict(nl=f"Sort '{filename}' {order} ({direction}){qualifiers}. Print the {end} {count} line(s).",
                           bash=pipeline, filename=filename, side=side, family=family,
                           ordering=[numeric, reverse, unique], count=count)


def rejected_command(row, sibling):
    """Select a reproducible one-edit alternative only if a fixture distinguishes it."""
    if fixture_result(row["bash"], row) is not True:
        raise ValueError(f"Reference fails its fixture: {row['nl']}")
    candidates = [sibling["bash"]]
    if "ordering" in row:
        sort, end = row["bash"].split(" | ")
        for flag in ("-n", "-r", "-u"):
            args = shlex.split(sort)
            if flag in args:
                args.remove(flag)
            else:
                args.insert(1, flag)
            candidates.append(shlex.join(args) + " | " + end)
        candidates.append(sort + " | " + end[:-1] + str(row["count"] % 3 + 1))
    # Rotate which semantic edit is tried first across filenames and requests.
    start = int.from_bytes(hashlib.sha256(row["nl"].encode()).digest()[:4], "big") % len(candidates)
    for cmd in candidates[start:] + candidates[:start]:
        if fixture_result(cmd, row) is False:
            return cmd
    raise ValueError(f"No discriminating fixture for {row['nl']}")


def fixture_cases(row):
    if "ordering" not in row:
        family = FAMILIES[row["family"]]
        return zip(family[2], family[row["side"]][2])
    numeric, reverse, unique = row["ordering"]
    cases = []
    for values in ((10, 2, 11, 2, 10, -3, 99, -3), (8, 8, 12, 100, -22, -1, 12)):
        lines = list(map(str, values))
        ordered = sorted(set(lines) if unique else lines,
                         key=int if numeric else str, reverse=reverse)
        n = row["count"]
        selected = ordered[-n:] if row["side"] else ordered[:n]
        cases.append(("\n".join(lines) + "\n", "\n".join(selected) + "\n"))
    return cases


def supported_filter(argv):
    if tuple(argv) in FILTERS:
        return True
    if argv[:1] == ["sort"]:
        return all(flag in ("-r", "-u", "-n") for flag in argv[1:])
    return (len(argv) == 3 and argv[0] in ("head", "tail")
            and argv[1] == "-n" and argv[2] in ("2", "3"))


def examples(families, files):
    for family in families:
        first, second, _ = FAMILIES[family]
        for filename in files:
            for side, (request, pipeline, _) in enumerate((first, second)):
                stages = pipeline.split(" | ")
                stages[0] += " " + shlex.quote(filename)
                yield dict(nl=request.format(file=filename), bash=" | ".join(stages),
                           family=family, filename=filename, side=side)


def fixture_result(command, row):
    """Return True, False, or None for syntax outside this evaluator's coverage."""
    filename = row["filename"]
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_. -]*", filename):
        raise ValueError("Invalid fixture filename")
    filters = []
    for i, stage in enumerate(command.split("|")):
        # Only literal words, options and the fixture operand are supported.
        if not re.fullmatch(r"[A-Za-z0-9_. '\t\"-]+", stage):
            return None
        try:
            argv = shlex.split(stage)
        except ValueError:
            return None
        if i == 0:
            if argv[-1:] != [filename]:
                return None
            argv.pop()
        if not supported_filter(argv):
            return None
        filters.append(argv)
    for stdin, expected in fixture_cases(row):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, filename).write_text(stdin, encoding="utf-8")
            output = ""
            for index, argv in enumerate(filters):
                args = [*argv, filename] if index == 0 else argv
                proc = subprocess.run(args, input=output, text=True, capture_output=True,
                                      cwd=directory, env={"PATH": os.defpath, "LC_ALL": "C"},
                                      timeout=2, check=True)
                output = proc.stdout
        if row["family"] == "counting":
            # The task asks for a count. Accept the optional fixture filename.
            output = output.strip().removesuffix(" " + filename) + "\n"
        if output != expected:
            return False
    return True
