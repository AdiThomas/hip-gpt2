#!/usr/bin/env python3
"""Turn solved kernel sources into exercises (SPEC section 5.1).

Everything strictly between a line containing only `// BEGIN SOLUTION` and a
line containing only `// END SOLUTION` (leading/trailing whitespace allowed)
is removed and replaced by one `// TODO: your code here` line, indented like
the BEGIN marker. The marker lines themselves and every other byte of the file
are preserved (line endings of the BEGIN line are reused for the TODO line).
Unbalanced or nested markers are an error (exit 1, nothing written for that
run's failing file; other errors are collected and reported).

Usage: make_exercises.py --src src --out /tmp/ex     (copies the whole tree,
       transforming *.hip *.cpp *.hpp *.h *.inc *.cu files)
       make_exercises.py --self-test
Standard library only.
"""
import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

BEGIN = re.compile(rb"^([ \t]*)//[ \t]*BEGIN SOLUTION[ \t]*(\r?\n?)$")
END = re.compile(rb"^[ \t]*//[ \t]*END SOLUTION[ \t]*\r?\n?$")
SUFFIXES = {".hip", ".cpp", ".hpp", ".h", ".inc", ".cu"}


class MarkerError(Exception):
    pass


def blank_solutions(data, label="<input>"):
    """Return transformed bytes; raise MarkerError on unbalanced/nested markers."""
    out = []
    open_line = None
    for lineno, line in enumerate(data.splitlines(keepends=True), 1):
        m_begin = BEGIN.match(line)
        if m_begin:
            if open_line is not None:
                raise MarkerError(f"{label}:{lineno}: nested BEGIN SOLUTION "
                                  f"(previous opened at line {open_line})")
            open_line = lineno
            indent, nl = m_begin.group(1), m_begin.group(2) or b"\n"
            out.append(line if line.endswith((b"\n", b"\r")) else line + nl)
            out.append(indent + b"// TODO: your code here" + nl)
            continue
        if END.match(line):
            if open_line is None:
                raise MarkerError(f"{label}:{lineno}: END SOLUTION without BEGIN SOLUTION")
            open_line = None
            out.append(line)
            continue
        if open_line is None:
            out.append(line)
    if open_line is not None:
        raise MarkerError(f"{label}:{open_line}: BEGIN SOLUTION never closed")
    return b"".join(out)


def process_tree(src, dst):
    src, dst = Path(src), Path(dst)
    errors, n_files, n_blanked = [], 0, 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        target = dst / p.relative_to(src)
        data = p.read_bytes()
        if p.suffix in SUFFIXES:
            try:
                new = blank_solutions(data, str(p))
            except MarkerError as exc:
                errors.append(str(exc))
                continue
            n_blanked += new != data
            data = new
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        shutil.copymode(p, target)
        n_files += 1
    return n_files, n_blanked, errors


FIXTURE = (b"// STAGE: v9 - fixture\n"
           b"constexpr int T = 4;\n"
           b"__global__ void k(float* x) {\n"
           b"    // BEGIN SOLUTION\n"
           b"    x[0] = 1.0f;\n"
           b"    if (x[1]) {\n"
           b"        x[2] = 2.0f;\n"
           b"    }\n"
           b"    // END SOLUTION\n"
           b"}\n"
           b"void launch_k(float* x) {\n"
           b"\t// BEGIN SOLUTION  \n"
           b"\tk(x);\n"
           b"\t// END SOLUTION\n"
           b"}\n"
           b"// trailing text without newline")

FIXTURE_EXPECTED = (b"// STAGE: v9 - fixture\n"
                    b"constexpr int T = 4;\n"
                    b"__global__ void k(float* x) {\n"
                    b"    // BEGIN SOLUTION\n"
                    b"    // TODO: your code here\n"
                    b"    // END SOLUTION\n"
                    b"}\n"
                    b"void launch_k(float* x) {\n"
                    b"\t// BEGIN SOLUTION  \n"
                    b"\t// TODO: your code here\n"
                    b"\t// END SOLUTION\n"
                    b"}\n"
                    b"// trailing text without newline")


def self_test():
    failures = []

    def check(name, cond):
        print(f"{'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    def raises(data):
        try:
            blank_solutions(data)
        except MarkerError:
            return True
        return False

    check("blanks body, keeps everything else byte-identical",
          blank_solutions(FIXTURE) == FIXTURE_EXPECTED)
    check("no markers -> unchanged", blank_solutions(b"int x;\r\nint y;") == b"int x;\r\nint y;")
    crlf = FIXTURE.replace(b"\n", b"\r\n")
    check("CRLF preserved", blank_solutions(crlf) == FIXTURE_EXPECTED.replace(b"\n", b"\r\n"))
    check("idempotent", blank_solutions(FIXTURE_EXPECTED) == FIXTURE_EXPECTED)
    check("marker as last line without newline",
          blank_solutions(b"// BEGIN SOLUTION\nx\n// END SOLUTION")
          == b"// BEGIN SOLUTION\n// TODO: your code here\n// END SOLUTION")
    check("marker text inside a longer line is ignored",
          blank_solutions(b"// see BEGIN SOLUTION markers\nint a;\n")
          == b"// see BEGIN SOLUTION markers\nint a;\n")
    check("nested BEGIN fails",
          raises(b"// BEGIN SOLUTION\n// BEGIN SOLUTION\n// END SOLUTION\n// END SOLUTION\n"))
    check("END without BEGIN fails", raises(b"int a;\n// END SOLUTION\n"))
    check("unclosed BEGIN fails", raises(b"// BEGIN SOLUTION\nint a;\n"))
    check("second END fails", raises(b"// BEGIN SOLUTION\n// END SOLUTION\n// END SOLUTION\n"))

    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        (src / "kernels").mkdir(parents=True)
        (src / "kernels" / "k.hip").write_bytes(FIXTURE)
        (src / "notes.txt").write_bytes(b"// BEGIN SOLUTION\nkeep me\n")
        n, blanked, errs = process_tree(src, out)
        check("tree: transformed and copied",
              not errs and n == 2 and blanked == 1
              and (out / "kernels" / "k.hip").read_bytes() == FIXTURE_EXPECTED
              and (out / "notes.txt").read_bytes() == b"// BEGIN SOLUTION\nkeep me\n")
        (src / "kernels" / "bad.hip").write_bytes(b"// BEGIN SOLUTION\n")
        n, blanked, errs = process_tree(src, Path(tmp) / "out2")
        check("tree: unbalanced file reported", len(errs) == 1 and "bad.hip:1" in errs[0]
              and not (Path(tmp) / "out2" / "kernels" / "bad.hip").exists())

    if failures:
        print(f"SELF-TEST FAILED: {failures}")
        return 1
    print("SELF-TEST OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="src")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.out:
        ap.error("--out is required")
    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        print(f"error: source directory not found: {src}", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()):
        print(f"error: output directory {out} is not empty", file=sys.stderr)
        return 2
    if out.resolve() == src.resolve() or src.resolve() in out.resolve().parents:
        print("error: --out must be outside --src", file=sys.stderr)
        return 2
    n, blanked, errors = process_tree(src, out)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"{n} files written to {out}, {blanked} with solutions blanked")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
