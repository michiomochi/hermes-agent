"""Static pre-screen for execute_code (``tools.approval_code_scan``).

Three corpora, all table-driven:

* SAFE — programs the allow-list recognises end to end.
* REVIEW / HIGH — programs that are not recognised, or that match a concrete high-risk pattern.
* ESCAPES — constructs an attacker (or a prompt-injected agent) would use to hide an effect from a
  *deny-list*. None of them may come out SAFE. When this list grows, the allow-list may have grown too.
"""

from __future__ import annotations

import os

import pytest

from tools.approval_code_scan import CodeScan, capability_key, capability_persistable, scan_python

ROOT = "@S@"


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    scratch = tmp_path / "scratch"
    work = tmp_path / "work"
    for d in (home, scratch, work):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (scratch / "linked").mkdir()

    class Env:
        pass

    e = Env()
    e.home, e.scratch, e.work = home, scratch, work
    e.hermes = home / ".hermes"

    def scan(code: str, deny=()) -> CodeScan:
        code = code.replace("@S@", str(scratch)).replace("@H@", str(home))
        return scan_python(code, write_roots=[scratch], cwd=work, deny_globs=list(deny), hermes_home=e.hermes)

    e.scan = scan
    return e


SAFE = {
    "print": 'print("hello")',
    "stdlib-math": "import json, re, math\nxs = [1, 2, 3]\nprint(json.dumps({'m': sum(xs) / len(xs), 'p': math.pi}))",
    "regex+collections": (
        "import re\nfrom collections import Counter, defaultdict\n"
        "words = re.findall(r'\\w+', 'a b a')\nprint(Counter(words).most_common(1))"
    ),
    "functions-classes": (
        "from dataclasses import dataclass\n\n@dataclass\nclass P:\n    x: int\n    y: int = 0\n\n"
        "def total(ps):\n    return sum(p.x + p.y for p in ps)\n\n"
        "try:\n    print(total([P(1), P(2, 3)]))\nexcept ValueError as e:\n    print(e)\n"
    ),
    "datetime": "import datetime\nprint(datetime.datetime.now().isoformat())",
    "local-method-call": "class A:\n    def f(self):\n        return 1\nprint(A().f())",
    "main-guard": "def main():\n    print(1)\n\nif __name__ == '__main__':\n    main()",
    "sqlite-readonly": (
        "import sqlite3\nconn = sqlite3.connect('@H@/.hermes/state.db')\n"
        "rows = conn.execute('SELECT id FROM sessions WHERE a = ?', (1,)).fetchall()\nprint(rows)"
    ),
    "sqlite-readonly-uri": (
        "import sqlite3\nconn = sqlite3.connect('file:@H@/data.db?mode=ro', uri=True)\n"
        "print(conn.execute('select 1').fetchone())"
    ),
    "sqlite-memory-write": (
        "import sqlite3\nc = sqlite3.connect(':memory:')\nc.execute('CREATE TABLE t(x)')\nc.execute('INSERT INTO t VALUES (1)')"
    ),
    "sqlite-write-under-root": (
        "import sqlite3\nc = sqlite3.connect('@S@/x.db')\nc.execute('CREATE TABLE t(x)')\nc.commit()"
    ),
    "write-scratch-open": "with open('@S@/out.txt', 'w') as f:\n    f.write('x')",
    "write-scratch-pathlib": "from pathlib import Path\nPath('@S@/out.txt').write_text('x')",
    "write-scratch-const-name": (
        "from pathlib import Path\nOUT = Path('@S@') / 'sub' / 'o.json'\nOUT.parent.mkdir(parents=True, exist_ok=True)\n"
        "OUT.write_text('{}')"
    ),
    "write-scratch-fstring": "base = '@S@'\nopen(f'{base}/a.txt', 'a').write('x')",
    "read-plain-file": "print(open('@H@/notes/todo.md').read())",
    "read-path-home": "from pathlib import Path\nprint((Path.home() / 'notes' / 'a.md').read_text())",
    "terminal-gated": "from hermes_tools import terminal\nr = terminal('ls -la')\nprint(r)",
    "terminal-via-module": "import hermes_tools\nprint(hermes_tools.terminal(f'echo {1}'))",
    "web-literal": "from hermes_tools import web_search, web_extract\nweb_search('hermes agent')\nweb_extract(['https://example.com/a'])",
    "hermes-file-tools-literal": (
        "from hermes_tools import read_file, write_file\nprint(read_file('@H@/a.txt'))\nwrite_file('@S@/b.txt', 'x')"
    ),
    "str-replace-two-args": "print('abc'.replace('a', 'b'))",
    "datetime-replace-keywords": "import datetime\nprint(datetime.datetime.now().replace(hour=0, minute=0))",
    "list-copy": "xs = [1, 2]\nprint(xs.copy())",
    "copy-within-roots": "from pathlib import Path\nPath('@H@/notes/a.md').copy('@S@/b.md')",
    "pragma-table-info": (
        "import sqlite3\nc = sqlite3.connect('@H@/.hermes/state.db')\n"
        "print(c.execute('PRAGMA table_info(sessions)').fetchall())"
    ),
    "pragma-user-version-bare": (
        "import sqlite3\nc = sqlite3.connect('@H@/.hermes/state.db')\nprint(c.execute('PRAGMA user_version').fetchone())"
    ),
    "self-method-call": (
        "class A:\n    def f(self):\n        return 1\n    def g(self):\n        return self.f()\nprint(A().g())"
    ),
    "local-instance-method": "class A:\n    def f(self):\n        return 1\na = A()\nprint(a.f())",
    "inherited-local-method": "class A:\n    def f(self):\n        return 1\nclass B(A):\n    pass\nprint(B().f())",
    "git-lookalike-name-in-roots": "from pathlib import Path\nPath('@S@/.gitignore').write_text('x')",
    "env-single-benign-key": "import os\nprint(os.environ['HOME'])",  # `os` itself is REVIEW (see below); kept here to document
}
SAFE.pop("env-single-benign-key")

REVIEW = {
    "import-os": "import os\nprint(os.listdir('.'))",
    "import-subprocess-literal-benign": "import subprocess\nsubprocess.run(['ls', '-la'])",
    "path-parameter": "def read(p):\n    return open(p).read()\nprint(read('a'))",
    "path-from-loop": "from pathlib import Path\nfor p in Path('@H@/docs').glob('*.md'):\n    print(p.read_text())",
    "write-outside-roots": "open('@H@/notes.txt', 'w').write('x')",
    "write-dynamic-mode": "m = input()\nopen('@S@/a', m)",
    "web-extract-fstring": "from hermes_tools import web_extract\ndef go(q):\n    web_extract([f'https://e.com/{q}'])\ngo('x')",
    "web-search-fstring-of-file-data": (
        "from hermes_tools import read_file, web_search\nweb_search(f\"x {read_file('@H@/a')}\")"
    ),
    "web-search-variable": "from hermes_tools import web_search\nimport json\nweb_search(json.dumps({}))",
    "other-tool": "from hermes_tools import search_files\nsearch_files('x')",
    "sqlite-dynamic-sql": "import sqlite3\nc = sqlite3.connect(':memory:')\nq = input()\nc.execute(q)",
    "sqlite-write-outside": "import sqlite3\nc = sqlite3.connect('@H@/state.db')\nc.execute('DELETE FROM t')",
    "sqlite-write-sneaky-cte": (
        "import sqlite3\nc = sqlite3.connect('@H@/state.db')\nc.execute('WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM a')"
    ),
    "sqlite-attach": "import sqlite3\nc = sqlite3.connect(':memory:')\nc.execute(\"ATTACH DATABASE '@H@/x.db' AS x\")",
    "syntax-error": "def f(:\n  pass",
    "unknown-name": "foo()",
    "unknown-method": "x = []\nx.explode()",
    "async": "async def f():\n    pass",
    "local-method-named-like-effect": "class A:\n    def write_text(self, x):\n        return x\nA().write_text(1)",
    "star-import": "from json import *\nprint(loads('1'))",
    "relative-import": "from . import x",
    "symlink": "from pathlib import Path\nPath('@S@/l').symlink_to('/etc/passwd')",
    "oversize": "x = 1\n" * 500,
    "match-statement": "match 1:\n    case 1:\n        pass",
}

HIGH = {
    "dotenv-literal": "print(open('@H@/.env').read())",
    "dotenv-pathlib-home": "from pathlib import Path\nprint((Path.home() / '.env').read_text())",
    "auth-json": "from pathlib import Path\nprint(Path.home().joinpath('.hermes', 'auth.json').read_text())",
    "ssh-key": "print(open('@H@/.ssh/id_rsa').read())",
    "etc-shadow": "print(open('/etc/shadow').read())",
    "expanduser-ssh": "from pathlib import Path\nprint(Path('~/.ssh/id_rsa').expanduser().read_text())",
    "tilde-auth-via-tool": "from hermes_tools import read_file\nprint(read_file('~/.hermes/auth.json'))",
    "proc-environ": "print(open('/proc/self/environ').read())",
    "token-named-file": "print(open('@H@/my_token.txt').read())",
    "environ-dump-print": "import os\nprint(dict(os.environ))",
    "environ-loop": "import os\nfor k, v in os.environ.items():\n    print(k, v)",
    "environ-secret-key": "import os\nprint(os.environ['OPENAI_API_KEY'])",
    "getenv-secret-key": "import os\nprint(os.getenv('GITHUB_TOKEN'))",
    "from-environ": "from os import environ\nprint(environ)",
    "subprocess-rm-rf": "import subprocess\nsubprocess.run('rm -rf ~', shell=True)",
    "os-system-rm": "import os\nos.system('rm -rf /home/x/project')",
    "subprocess-list-form": "import subprocess\nsubprocess.run(['rm', '-rf', '/home/x/project'])",
    "rmtree-outside": "import shutil\nshutil.rmtree('@H@/project')",
    "rmtree-dynamic": "import shutil\nimport json\nshutil.rmtree(json.loads('1'))",
    "unlink-outside": "from pathlib import Path\nPath('@H@/important.txt').unlink()",
    "unlink-unknown-receiver": "def rm(p):\n    p.unlink()\nrm(1)",
    "base64-exec": "import base64\nexec(base64.b64decode('cHJpbnQoMSk='))",
    "decode-then-eval-via-var": "import codecs\ns = codecs.decode('cevag(1)', 'rot13')\neval(s)",
    "write-hermes-config": "open('@H@/.hermes/config.yaml', 'a').write('x')",
    "write-hermes-source": "from pathlib import Path\nPath.home().joinpath('.hermes', 'hermes-agent', 'tools', 'approval.py').write_text('')",
    "sqlite-read-cred": "import sqlite3\nsqlite3.connect('@H@/.hermes/auth.json').execute('select 1')",
}

ESCAPES = {
    "getattr-concat": "getattr(__builtins__, 'ev' + 'al')('1')",
    "alias-open-then-write": "o = open\no('@S@/a', 'w')",
    "alias-open-passed": "list(map(open, ['@H@/a']))",
    "alias-path-method": "from pathlib import Path\nw = Path('@S@/a').write_text\nw('x')",
    "alias-path-method-partial": "from pathlib import Path\nimport functools\nfunctools.partial(Path.write_text, Path('@H@/a'), 'x')()",
    "dunder-import": "__import__('os').system('id')",
    "sys-modules": "import sys\nsys.modules['os'].system('id')",
    "sys-unimported": "sys.modules",
    "lambda-default-arg": "f = lambda x=getattr: x\nf()(1, 'real')",
    "decorator-exec": "@exec\ndef f():\n    pass",
    "decorator-lambda": "@(lambda f: f)\ndef f():\n    pass",
    "type-call": "type('X', (), {})",
    "type-bases": "T = type\nT('X', (object,), {'x': 1})",
    "unicode-fullwidth-eval": "\uff45\uff56\uff41\uff4c('1')",
    "unicode-cyrillic-eval": "\u0435val('1')",
    "auth-json-via-home": "from pathlib import Path\nprint((Path.home() / '.hermes' / 'auth.json').read_text())",
    "env-via-concat": "print(open('.e' + 'nv').read())",
    "env-via-concat-dynamic": "x = input()\nprint(open('.e' + x).read())",
    "dunder-class": "[].__class__.__mro__",
    "dunder-globals": "(lambda: 0).__globals__",
    "dunder-in-string": "'{0.__init__.__globals__}'.format(print)",
    "module-attr-leak": "import pathlib\npathlib.os.system('id')",
    "module-attr-leak-json": "import json\njson.codecs",
    "module-as-value": "import json\nj = json\nj.codecs",
    "get-type-hints-eval": "from typing import get_type_hints\nget_type_hints(print)",
    "string-formatter": "import string\nstring.Formatter().get_field('0.__class__', (1,), {})",
    "metaclass": "class A(metaclass=print):\n    pass",
    "class-base-unknown": "class A(unknown):\n    pass",
    "sqlite-extension": "import sqlite3\nc = sqlite3.connect(':memory:')\nc.enable_load_extension(True)",
    "open-fd": "open(0)",
    "open-opener": "open('@S@/a', opener=print)",
    "open-star-kwargs": "d = {}\nopen('@S@/a', **d)",
    "globals-call": "globals()",
    "compile-call": "compile('1', 'x', 'eval')",
    "breakpoint": "breakpoint()",
    "input-read": "input()",
    "rebound-global-path": (
        "p = '@S@/a'\ndef f():\n    global p\n    p = '@H@/.bashrc'\nf()\nopen(p, 'w')"
    ),
    "rebound-by-augassign": "p = '@S@/a'\np += '../../x'\nopen(p, 'w')",
    "rebound-in-loop": "p = '@S@/a'\nfor p in ['@H@/.bashrc']:\n    pass\nopen(p, 'w')",
    "shadow-open": "open = print\nopen('@H@/.bashrc', 'w')",
    "dotdot-escape": "open('@S@/../x', 'w')",
    "symlink-escape": "import os\n",  # placeholder replaced below by a real symlink test
    "tilde-is-not-guessed": "from hermes_tools import write_file\nwrite_file('~/notes.txt', 'x')",
    "file-dunder": "print(__file__)",
    "walrus-path": "open((p := '@H@/.bashrc'), 'w')",
    "write-bytes-outside": "from pathlib import Path\nPath('@H@/x').write_bytes(b'')",
    "replace-path-one-arg": "from pathlib import Path\nPath('@S@/a').replace('@H@/b')",
    "unbound-call-of-call": "f = lambda: print\nf()('x')",
    "global-stmt-import-shadow": "import json\njson = 1\njson.dumps",
    "print-file-kwarg-open": "print('x', file=open('@H@/a', 'w'))",
    "exec-in-string-eval-name": "e = eval\ne('1')",
    "nested-import-in-function": "def f():\n    import os\n    return os.getcwd()",
    "importlib": "import importlib\nimportlib.import_module('os')",
    "pickle": "import pickle\npickle.loads(b'')",
    "ctypes": "import ctypes",
    "socket": "import socket",
    "urllib": "import urllib.request",
    "attribute-store-on-module": "import json\njson.dumps = print",
    "yield-exec": "def g():\n    yield eval\n",
}
ESCAPES.pop("symlink-escape")


@pytest.mark.parametrize("name", sorted(SAFE))
def test_safe_corpus(env, name):
    scan = env.scan(SAFE[name])
    assert scan.tier == "safe", (scan.reasons, sorted(scan.features))


@pytest.mark.parametrize("name", sorted(REVIEW))
def test_review_corpus(env, name):
    scan = env.scan(REVIEW[name])
    assert scan.tier == "review", (scan.tier, scan.reasons)


@pytest.mark.parametrize("name", sorted(HIGH))
def test_high_corpus(env, name):
    scan = env.scan(HIGH[name])
    assert scan.tier == "high", (scan.tier, scan.reasons, sorted(scan.features))


@pytest.mark.parametrize("name", sorted(ESCAPES))
def test_escape_corpus_is_never_safe(env, name):
    scan = env.scan(ESCAPES[name])
    assert scan.tier != "safe", (name, sorted(scan.features))


def test_symlink_inside_root_pointing_out_is_not_a_root_write(env):
    target = env.home / "elsewhere"
    target.mkdir()
    os.symlink(target, env.scratch / "jump")
    scan = env.scan("open('@S@/jump/a.txt', 'w').write('x')")
    assert scan.tier != "safe", scan.reasons


def test_symlink_to_credential_file_is_high(env):
    secret = env.home / ".env"
    secret.write_text("K=v")
    os.symlink(secret, env.scratch / "innocent.txt")
    scan = env.scan("print(open('@S@/innocent.txt').read())")
    assert scan.tier == "high", scan.reasons


def test_relative_dotenv_is_resolved_against_cwd(env):
    assert env.scan("print(open('.env').read())").tier == "high"


def test_write_under_cwd_is_not_a_root_unless_configured(env):
    assert env.scan("open('out.txt', 'w').write('x')").tier == "review"
    scan = scan_python("open('out.txt', 'w').write('x')", write_roots=[env.work], cwd=env.work, deny_globs=[],
                       hermes_home=env.hermes)
    assert scan.tier == "safe"


def test_filesystem_root_is_never_a_write_root(env):
    scan = scan_python("open('/tmp/x', 'w').write('x')", write_roots=[os.sep], cwd=env.work, deny_globs=[],
                       hermes_home=env.hermes)
    assert scan.tier == "review"


def test_deny_glob_applies_to_paths_and_calls(env):
    code = "print(open('@S@/secrets-list.txt').read())"
    assert env.scan(code).tier == "safe"
    assert env.scan(code, deny=["*secrets-list*"]).tier == "high"
    # approvals.deny also covers the stringified call, e.g. a command text inside a hermes_tools call.
    call = "from hermes_tools import terminal\nterminal('git push --force origin main')"
    assert env.scan(call).tier == "safe"
    assert env.scan(call, deny=["*git push --force*"]).tier == "high"


def test_deny_glob_matches_shell_commands_given_to_subprocess(env):
    scan = env.scan("import subprocess\nsubprocess.run('git push --force', shell=True)", deny=["git push --force*"])
    assert scan.tier == "high"


def test_high_reasons_name_the_line(env):
    scan = env.scan("x = 1\nprint(open('@H@/.env').read())")
    assert any("line 2" in r for r in scan.reasons)


def test_never_raises_on_garbage(env):
    for code in ("\x00", "(" * 5000, "x = " + "[" * 400 + "]" * 400, "\udcff = 1"):
        scan = env.scan(code)
        assert isinstance(scan, CodeScan) and scan.tier != "safe"


def test_syntax_error_and_oversize_are_per_script_keys(env):
    a, b = env.scan("def f(:"), env.scan("def g(:")
    assert capability_key(a, "execute_code") != capability_key(b, "execute_code")
    big_a, big_b = env.scan("x = 1\n" * 500), env.scan("y = 1\n" * 500)
    assert capability_key(big_a, "execute_code") != capability_key(big_b, "execute_code")


# --- capability_key ---------------------------------------------------------------------------------------------

def test_capability_key_is_stable_and_namespaced(env):
    scan = env.scan("print(1)")
    assert capability_key(scan, "execute_code") == "execute_code:cap:net:none|proc:none|write:none"
    assert capability_key(scan, "python_inline").startswith("python_inline:cap:")


def test_capability_key_collapses_each_axis_to_its_maximum(env):
    key = capability_key(CodeScan("review", frozenset({"write:roots", "write:outside", "net:fixed"}), ()), "ns")
    assert key == "ns:cap:net:fixed|proc:none|write:outside"


def test_capability_key_distinguishes_capabilities(env):
    plain = capability_key(env.scan("print(1)"), "execute_code")
    with_os = capability_key(env.scan("import os\nprint(os.listdir('.'))"), "execute_code")
    with_kill = capability_key(env.scan("import os\nos.kill(1, 9)"), "execute_code")
    with_proc = capability_key(env.scan("import subprocess\nsubprocess.run(['ls'])"), "execute_code")
    assert len({plain, with_os, with_kill, with_proc}) == 4
    assert "import:os" in with_os and "call:os.kill" in with_kill and "proc:direct" in with_proc


def test_review_grant_for_one_unrecognised_name_does_not_cover_another(env):
    a = capability_key(env.scan("foo()"), "execute_code")
    b = capability_key(env.scan("bar()"), "execute_code")
    assert a != b


# --- review round 2: one reproduction per hole ------------------------------------------------------------------
# Each test names an input that an earlier revision rated SAFE although it changes something outside the write roots.

def _scan_with_roots(env, code, roots, cwd=None):
    code = code.replace("@S@", str(env.scratch)).replace("@H@", str(env.home)).replace("@W@", str(env.work))
    return scan_python(code, write_roots=roots, cwd=cwd or env.work, deny_globs=[], hermes_home=env.hermes)


@pytest.mark.parametrize("code, tier", [
    ("from pathlib import Path\nPath('~/.hermes/config.yaml').rename(target='@S@/x')", "high"),
    ("from pathlib import Path\nPath('@S@/a').rename(target='@H@/b')", "review"),
    ("from pathlib import Path\nPath('@S@/a').replace(target='@H@/b')", "review"),
    ("from pathlib import Path\nPath('@S@/a').rename('@H@/b')", "review"),
    ("from pathlib import Path\nd = {}\nPath('@S@/a').rename(**d)", "review"),
])
def test_blocker1_path_rename_and_replace_with_keyword_target_is_a_move(env, code, tier):
    scan = env.scan(code)
    assert scan.tier == tier, (scan.tier, scan.reasons, sorted(scan.features))


@pytest.mark.parametrize("sql", [
    "PRAGMA schema_version(1)", "PRAGMA journal_mode(OFF)", "PRAGMA user_version(5)", "PRAGMA wal_checkpoint",
    "PRAGMA optimize", "PRAGMA main.schema_version(1)",
])
def test_blocker2_pragma_with_argument_or_side_effect_is_a_db_write(env, sql):
    scan = env.scan(f"import sqlite3\nc = sqlite3.connect('@H@/.hermes/state.db')\nc.execute({sql!r})")
    assert scan.tier == "review" and "db:write" in scan.features, (scan.reasons, sorted(scan.features))


@pytest.mark.parametrize("target", [
    "file:@S@/%2E%2E/x.db", "file:@S@/%2e%2e/%2e%2e/x.db", "file://localhost@S@/%2E%2E/x.db",
])
def test_blocker3_percent_encoded_dotdot_in_a_sqlite_uri_leaves_the_roots(env, target):
    scan = env.scan(f"import sqlite3\nc = sqlite3.connect({target!r}, uri=True)\nc.execute('CREATE TABLE t(x)')")
    assert scan.tier == "review" and "db:write" in scan.features, (scan.reasons, sorted(scan.features))


@pytest.mark.parametrize("uri_arg", ["uri=1", "uri='yes'", "uri=flag"])
def test_blocker3_truthy_or_dynamic_uri_flag_is_not_treated_as_a_plain_path(env, uri_arg):
    """With the cwd as a write root, the plain-path reading puts ``file:/...`` under the cwd; SQLite opens ``/...``."""
    code = (f"import json\nimport sqlite3\nflag = json.loads('1')\nc = sqlite3.connect('file:@H@/.hermes/state.db', {uri_arg})\n"
            "c.execute('CREATE TABLE t(x)')")
    scan = _scan_with_roots(env, code, [env.work])
    assert scan.tier != "safe", (scan.reasons, sorted(scan.features))


def test_blocker3_file_uri_with_an_unknown_uri_flag_is_review_even_for_reads(env):
    scan = env.scan("import json\nimport sqlite3\nflag = json.loads('1')\nsqlite3.connect('file:@H@/a.db', uri=flag).execute('select 1')")
    assert scan.tier == "review"


def test_blocker3_positional_uri_flag_is_honoured(env):
    code = ("import sqlite3\nc = sqlite3.connect('file:@S@/%2E%2E/x.db', 5.0, 0, 'DEFERRED', True, None, 128, True)\n"
            "c.execute('CREATE TABLE t(x)')")
    assert env.scan(code).tier == "review"
    # Same program with the cwd as a root and a plain absolute target: still decoded, still outside.
    assert _scan_with_roots(env, code.replace("@S@/%2E%2E/x.db", "@H@/%2E%2E/x.db"), [env.work]).tier == "review"


@pytest.mark.parametrize("call", [
    "Path('@S@/a').copy('@H@/.bashrc')", "Path('@S@/a').copy(target='@H@/.bashrc')", "Path('@S@/a').copy_into('@H@')",
    "Path('@S@/a').move('@H@/.bashrc')", "Path('@S@/a').move_into('@H@')", "Path('@H@/a').move('@S@/b')",
    "Path('@S@/a').copy(**{})",
])
def test_blocker4_path_copy_and_move_are_path_operations(env, call):
    scan = env.scan(f"from pathlib import Path\n{call}")
    assert scan.tier != "safe", (scan.reasons, sorted(scan.features))


def test_blocker4_path_copy_to_hermes_config_is_high(env):
    scan = env.scan("from pathlib import Path\nPath('@S@/a').copy('~/.hermes/config.yaml')")
    assert scan.tier == "high", (scan.reasons, sorted(scan.features))


def test_blocker4_aliasing_copy_or_replace_escapes_the_vetting(env):
    assert env.scan("from pathlib import Path\nf = Path('@S@/a').copy\nf('@H@/.bashrc')").tier != "safe"
    assert env.scan("from pathlib import Path\nf = Path('@S@/a').replace\nf('@H@/.bashrc')").tier != "safe"


def test_blocker5_a_method_defined_on_some_class_does_not_whitelist_that_name_on_other_receivers(env):
    code = (
        "import sqlite3\n"
        "class Mine:\n    def backup(self, target):\n        return target\n"
        "mem = sqlite3.connect(':memory:')\n"
        "mem.backup(sqlite3.connect('@H@/.hermes/state.db'))\n"
    )
    scan = env.scan(code)
    assert scan.tier == "review" and "call:method:backup" in scan.features, (scan.reasons, sorted(scan.features))


def test_blocker5_unknown_receiver_calling_a_locally_defined_method_name_is_review(env):
    scan = env.scan("class A:\n    def run(self):\n        return 1\ndef make():\n    return input()\nmake().run()")
    assert scan.tier == "review"


def test_blocker5_a_local_class_does_not_whitelist_its_method_for_another_classs_instance(env):
    scan = env.scan("class A:\n    def f(self):\n        return 1\nclass B:\n    pass\nB().f()")
    assert scan.tier == "review"


@pytest.mark.parametrize("name", [".git/hooks/pre-commit", ".git/config", "sub/.git", ".git"])
def test_blocker7_git_metadata_inside_a_write_root_is_not_a_root_write(env, name):
    scan = env.scan(f"from pathlib import Path\nPath('@S@/{name}').write_text('x')")
    assert scan.tier == "review" and "write:outside" in scan.features, (scan.reasons, sorted(scan.features))


def test_blocker7_git_dir_in_a_worktree_root_is_excluded_via_open_sqlite_and_delete(env):
    (env.work / ".git").write_text("gitdir: /elsewhere\n")
    roots = [env.work]
    assert _scan_with_roots(env, "open('@W@/.git', 'w').write('x')", roots).tier != "safe"
    assert _scan_with_roots(env, "open('@W@/.GIT/hooks/post-checkout', 'w').write('x')", roots).tier != "safe"
    assert _scan_with_roots(
        env, "import sqlite3\nsqlite3.connect('@W@/.git/x.db').execute('CREATE TABLE t(x)')", roots).tier != "safe"
    assert _scan_with_roots(env, "from pathlib import Path\nPath('@W@/.git/index').unlink()", roots).tier == "high"
    # ...while an ordinary file in the same root is still a root write.
    assert _scan_with_roots(env, "open('@W@/ok.txt', 'w').write('x')", roots).tier == "safe"


def test_blocker7_git_dir_reached_through_a_symlink_inside_the_root_is_excluded(env):
    (env.work / ".git").mkdir()
    os.symlink(env.work / ".git", env.work / "innocent")
    assert _scan_with_roots(env, "open('@W@/innocent/config', 'w').write('x')", [env.work]).tier != "safe"


# --- should-fix: credential globs, key shape --------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "@H@/.config/gh/hosts.yml", "@H@/.kube/config", "@H@/.docker/config.json", "@H@/.gnupg/private-keys-v1.d/a.key",
    "~/.config/gh/hosts.yml",
])
def test_more_credential_paths_are_high(env, path):
    assert env.scan(f"print(open({path!r}).read())").tier == "high"


@pytest.mark.parametrize("path", [
    "@H@/.bashrc", "@H@/.zshrc", "@H@/.config/app/settings.toml", "@H@/.local/share/x", "~/.bashrc",
])
def test_reading_dotfiles_directly_under_home_is_review(env, path):
    scan = env.scan(f"print(open({path!r}).read())")
    assert scan.tier == "review", (scan.reasons, sorted(scan.features))


def test_reading_under_hermes_home_and_plain_home_files_stays_safe(env):
    assert env.scan("print(open('@H@/.hermes/skills/x/SKILL.md').read())").tier == "safe"
    assert env.scan("print(open('@H@/notes/.hidden').read())").tier == "safe"


def test_capability_key_over_512_chars_is_rounded_to_an_opaque_digest(env):
    features = frozenset(f"call:mod{i}.function_with_a_long_name_{i}" for i in range(40))
    key = capability_key(CodeScan("review", features, ()), "execute_code")
    assert len(key) <= 512 and key.startswith("execute_code:cap:opaque:")
    assert key != capability_key(CodeScan("review", features | {"call:other"}, ()), "execute_code")
    assert key == capability_key(CodeScan("review", features, ()), "execute_code")


def test_capability_key_just_under_the_limit_is_kept_verbatim(env):
    key = capability_key(CodeScan("review", frozenset({"call:os.kill"}), ()), "execute_code")
    assert "opaque:" not in key and "call:os.kill" in key


@pytest.mark.parametrize("code", [
    "open('@H@/notes.txt', 'w').write('x')",                                     # write:outside
    "def r(p):\n    return open(p).read()\nr('a')",                               # read:dynamic
    "import subprocess\nsubprocess.run(input())",                                 # proc:direct, command not constant
    "import subprocess\nsubprocess.run(['ls', input()])",                         # proc:direct, argv not constant
])
def test_coarse_capability_keys_are_not_persistable(env, code):
    scan = env.scan(code)
    assert scan.tier == "review" and not capability_persistable(scan), (scan.reasons, sorted(scan.features))


@pytest.mark.parametrize("code", [
    "import subprocess\nsubprocess.run(['ls', '-la'])",
    "import os\nprint(os.listdir('.'))",
    "from hermes_tools import web_search\nweb_search('x')",
    "open('@S@/a', 'w').write('x')",
])
def test_fine_grained_keys_stay_persistable(env, code):
    assert capability_persistable(env.scan(code))


def test_blocker3_percent_encoded_uri_path_is_never_safe(env):
    """SQLite percent-decodes the URI path, so a body can look in-root in the source text and
    still name a different file. The scanner refuses to reason about an encoded path at all:
    it must be REVIEW even when the decoded target happens to land inside a write root and the
    SQL is read-only."""
    scan = env.scan(
        "import sqlite3\n"
        "c = sqlite3.connect('file:@S@/%2E/x.db', uri=True)\n"
        "print(c.execute('SELECT 1').fetchone())\n"
    )
    assert scan.tier == "review" and "read:dynamic" in scan.features, (scan.reasons, sorted(scan.features))
