"""Deterministic repairs for the JSON that LLMs return to the dreaming pipeline.

Observed 2026-10-09 on the real nightly prompt: the chunking prompt makes the model copy raw conversation
text into JSON strings. That text contains backslashes (shell commands, Windows paths, regexes), and the
model emits them unescaped -- `Invalid \\escape` -- so strict parsing failed. The chunker's last-ditch
fallback then accepted the first inner list it could decode (an `entities` array of strings) as "chunks",
produced zero chunks, and the run still reported success. The same reply, repaired, is 18 valid chunks.
"""
import re

# A valid JSON escape (consumed as a pair so `\\\\` is never split) or a lone backslash.
_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def repair_json_escapes(text: str) -> str:
    """Double every backslash that does not start a valid JSON escape. Valid JSON is returned unchanged.

    Limit: a backslash followed by b, f, n, r, t or u+4 hex digits IS a valid escape, so `\\notes` in a path
    still decodes as a newline + "otes". Only the invalid cases (`\\U`, `\\d`, `\\.` ...) are repairable.
    """
    return _ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "\\\\", text)


def only_objects(items) -> bool:
    """True for a list whose entries are all JSON objects (an empty list qualifies).

    A list of strings or numbers is not a list of chunks/clusters/units, however it was reached.
    """
    return isinstance(items, list) and all(isinstance(x, dict) for x in items)
