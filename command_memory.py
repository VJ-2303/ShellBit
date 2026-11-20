"""Transfer observed flag edits between closely matching, read-only command requests."""

import json
import re
import shlex
from collections import defaultdict

from intent_pairs import supported_filter


# An operand must be explicitly quoted in the request and a literal filename.
_FILENAME = re.compile(r"""(['"])([A-Za-z0-9_][A-Za-z0-9_. -]*\.[A-Za-z0-9]+)\1""")
_SLOT = "<file>"


def request_key(request):
    matches = list(_FILENAME.finditer(request))
    if len(matches) != 1 or _SLOT in request.casefold():
        return None
    match = matches[0]
    text = request[:match.start()] + _SLOT + request[match.end():]
    return tuple(re.findall(r"<file>|\w+|[^\w\s]", text.casefold())), match[2]


def pipeline_shape(command, filename):
    stages = []
    for i, stage in enumerate(command.split("|")):
        if not re.fullmatch(r"""[A-Za-z0-9_. '"\t-]+""", stage):
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
        if argv[0] == "sort":
            argv = ["sort", *sorted(set(argv[1:]))]
        stages.append(tuple(argv))
    return tuple(stages)


def flag_edit(before, after):
    if len(before) != len(after):
        return None
    changed = [i for i in range(len(before)) if before[i] != after[i]]
    if len(changed) != 1:
        return None
    i = changed[0]
    # These supported sort switches commute. Order-sensitive arguments in
    # other stages remain literal, even if they resemble flags.
    if before[i][0] != "sort" or after[i][0] != "sort":
        return None
    removed = set(before[i]) - set(after[i])
    added = set(after[i]) - set(before[i])
    if len(removed) + len(added) != 1:
        return None
    return i, tuple(removed), tuple(added)


def context(words, pos, replacement, stages):
    return (words[max(0, pos-1):pos+2], pos == 0, pos == len(words)-1,
            replacement, tuple(stage[0] for stage in stages))


class CommandMemory:
    """Index literal matches and one-word contrasts using training prompts only."""

    def __init__(self, rows):
        entries = defaultdict(set)
        for row in rows:
            parsed = request_key(row["nl"])
            if parsed:
                words, filename = parsed
                stages = pipeline_shape(row["bash"], filename)
                if stages:
                    entries[words].add(stages)
        self.entries = entries
        self.neighbors = defaultdict(list)
        self.edits = defaultdict(lambda: defaultdict(set))
        for words in entries:
            for i in range(len(words)):
                self.neighbors[i, words[:i], words[i+1:]].append(words)
        for (i, _, _), group in self.neighbors.items():
            for source in group:
                for dest in group:
                    if source == dest:
                        continue
                    for before in entries[source]:
                        key = context(source, i, dest[i], before)
                        for after in entries[dest]:
                            edit = flag_edit(before, after)
                            self.edits[key][edit].add(source)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls(json.loads(line) for line in stream if line.strip())

    def translate(self, request):
        parsed = request_key(request)
        if not parsed:
            return None
        words, filename = parsed
        outputs = set(self.entries.get(words, ()))
        if not outputs:
            for i, word in enumerate(words):
                for source in self.neighbors.get((i, words[:i], words[i+1:]), ()):
                    for stages in self.entries[source]:
                        edits = self.edits.get(context(source, i, word, stages), {})
                        if len(edits) != 1:
                            continue
                        edit, witnesses = next(iter(edits.items()))
                        if edit is None or len(witnesses) < 2:
                            continue
                        index, removed, added = edit
                        flags = set(stages[index][1:])
                        if not flags.issuperset(removed) or flags.intersection(added):
                            continue
                        flags.difference_update(removed)
                        flags.update(added)
                        new_stage = ("sort", *sorted(flags))
                        outputs.add(stages[:index] + (new_stage,) + stages[index+1:])
        if len(outputs) != 1:
            return None
        stages = next(iter(outputs))
        return " | ".join(shlex.join((*stage, filename) if i == 0 else stage)
                          for i, stage in enumerate(stages))
