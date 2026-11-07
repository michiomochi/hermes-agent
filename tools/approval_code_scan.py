"""Static pre-screen for Python programs run by ``execute_code``.

:func:`scan_python` classifies a program into one of three tiers *without running it*:

``safe``
    Every construct in the program is on an allow-list (syntax, imports, names, calls, paths) and
    every file path it touches can be folded to a constant. Nothing here is a blacklist: a program
    is SAFE only because everything in it is *recognised*, never because nothing bad was found.
``high``
    A concrete high-risk pattern was found (credential read, ``os.environ`` dump, deletion outside
    the write roots, decoded-then-executed strings, a write to Hermes's own config/source, a
    shell command that the existing detectors or ``approvals.deny`` flag). Callers must put a
    human in the loop and must not persist the answer.
``review``
    Anything else: the caller keeps the existing behaviour (guardian LLM, then a human).

The scan can only ever *grant* a one-shot approval (SAFE) or *raise* the severity (HIGH). It never
lowers a guardian DENY/ESCALATE and is not a security boundary (SECURITY.md §2.2) — it is a cheap,
deterministic filter in front of the LLM. Any internal failure degrades to ``review``.

``features`` describe what the program can do (``net:none``, ``write:roots``, ``proc:gated`` …);
:func:`capability_key` folds them into an approval key so a "session" grant covers a capability
class instead of every script.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, Sequence
from urllib.parse import parse_qsl, unquote

logger = logging.getLogger("tools.approval")

Tier = Literal["safe", "review", "high"]

MAX_SAFE_LINES = 400
MAX_SAFE_BYTES = 32 * 1024
_MAX_SCAN_BYTES = 1024 * 1024
_MAX_REASONS = 50
_DENY_UNPARSE_LIMIT = 64 * 1024


@dataclass(frozen=True)
class CodeScan:
    tier: Tier
    features: frozenset[str]
    reasons: tuple[str, ...]


# --- Capability axes ----------------------------------------------------------------------------------------------
# Ordered low -> high. A signature keeps ONE value per axis (the highest seen); features that are not on an axis
# (``import:os``, ``call:os.kill``, ``name:unknown:x`` …) are kept verbatim, so a grant for one never covers another.
_AXES: dict[str, tuple[str, ...]] = {
    "net": ("none", "fixed", "dynamic"),
    "write": ("none", "roots", "outside", "dynamic", "symlink"),
    "read": ("none", "fs", "dynamic", "cred"),
    "db": ("none", "read", "write"),
    "del": ("none", "roots", "outside", "dynamic"),
    "proc": ("none", "gated", "direct"),
    "obf": ("none", "dynamic"),
    "env": ("none", "key", "dump"),
}
_ALWAYS_IN_SIGNATURE = ("net", "write", "proc")
_MAX_KEY_SIGNATURE = 512


def capability_key(scan: CodeScan, namespace: str) -> str:
    """Approval key for the capability class of ``scan``: ``<namespace>:cap:<signature>``."""
    best: dict[str, int] = {}
    loose: set[str] = set()
    for feature in scan.features:
        axis, _, value = feature.partition(":")
        order = _AXES.get(axis)
        if order is not None and value in order:
            best[axis] = max(best.get(axis, 0), order.index(value))
        else:
            loose.add(feature)
    for axis in _ALWAYS_IN_SIGNATURE:
        best.setdefault(axis, 0)
    parts = {f"{axis}:{_AXES[axis][idx]}" for axis, idx in best.items() if idx or axis in _ALWAYS_IN_SIGNATURE}
    signature = "|".join(sorted(parts | loose))
    if len(signature) > _MAX_KEY_SIGNATURE:
        # A program that names hundreds of distinct things must not store a huge key in the config.
        signature = "opaque:" + hashlib.sha256(signature.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{namespace}:cap:{signature}"


def capability_persistable(scan: CodeScan) -> bool:
    """False when the capability key is too coarse for a session/always grant to be safe.

    ``write:outside`` (and the dynamic/symlink writes above it), ``read:dynamic`` and ``proc:direct`` with a command
    that is not a constant each cover an unbounded set of targets under ONE key, so a single approval would pass all
    of them. Those are decided per run; every other key is as fine-grained as it ever was.
    """
    best: dict[str, int] = {}
    for feature in scan.features:
        axis, _, value = feature.partition(":")
        order = _AXES.get(axis)
        if order is not None and value in order:
            best[axis] = max(best.get(axis, 0), order.index(value))
    if best.get("write", 0) >= _AXES["write"].index("outside"):
        return False
    if best.get("read", 0) >= _AXES["read"].index("dynamic"):
        return False
    return "proc:nonconst" not in scan.features


def scan_summary(scan: CodeScan) -> dict:
    """Plain-dict form of a scan for the guardian prompt (the caller clips and JSON-encodes it)."""
    return {"tier": scan.tier, "features": sorted(scan.features), "reasons": list(scan.reasons)}


# --- Allow-lists --------------------------------------------------------------------------------------------------
# module -> attributes that may be reached directly on it. ``None`` = any public (non-underscore) attribute; only for
# C modules that import nothing else. Python modules get explicit lists because ``pathlib.os`` / ``json.codecs``
# would otherwise hand out ``os.system``.
_ANY: None = None
_MODULE_ATTRS: dict[str, frozenset[str] | None] = {
    "json": frozenset("loads load dumps dump JSONDecodeError JSONEncoder JSONDecoder".split()),
    "re": frozenset(
        "compile match search fullmatch findall finditer sub subn split escape error Pattern Match "
        "IGNORECASE I MULTILINE M DOTALL S VERBOSE X ASCII A UNICODE U".split()
    ),
    "math": _ANY,
    "statistics": frozenset(
        "mean fmean median median_low median_high median_grouped mode multimode stdev pstdev variance pvariance "
        "quantiles geometric_mean harmonic_mean correlation covariance linear_regression NormalDist "
        "StatisticsError".split()
    ),
    "collections": frozenset(
        "namedtuple deque Counter OrderedDict defaultdict ChainMap UserDict UserList UserString abc".split()
    ),
    "itertools": _ANY,
    "functools": frozenset(
        "reduce partial partialmethod lru_cache cache wraps cmp_to_key total_ordering singledispatch "
        "cached_property".split()
    ),
    "datetime": frozenset("datetime date time timedelta timezone tzinfo MINYEAR MAXYEAR UTC".split()),
    "time": frozenset(
        "time sleep perf_counter monotonic strftime strptime localtime gmtime mktime time_ns perf_counter_ns "
        "monotonic_ns ctime asctime process_time timezone altzone daylight tzname struct_time".split()
    ),
    "textwrap": frozenset("dedent indent wrap fill shorten TextWrapper".split()),
    "csv": frozenset(
        "reader writer DictReader DictWriter QUOTE_ALL QUOTE_MINIMAL QUOTE_NONNUMERIC QUOTE_NONE Dialect excel "
        "excel_tab unix_dialect field_size_limit Error Sniffer register_dialect".split()
    ),
    "string": frozenset(
        "ascii_letters ascii_lowercase ascii_uppercase digits hexdigits octdigits punctuation printable whitespace "
        "capwords Template".split()
    ),
    "dataclasses": frozenset(
        "dataclass field fields asdict astuple replace is_dataclass KW_ONLY MISSING FrozenInstanceError".split()
    ),
    "typing": frozenset(
        "Any Optional Union List Dict Set FrozenSet Tuple Callable Iterable Iterator Sequence Mapping MutableMapping "
        "Generator Type TypeVar Generic NamedTuple TypedDict Literal Final ClassVar Protocol cast overload "
        "TYPE_CHECKING Annotated NoReturn Self Never".split()
    ),
    "pprint": frozenset("pprint pformat PrettyPrinter pp saferepr isreadable isrecursive".split()),
    "hashlib": frozenset(
        "md5 sha1 sha224 sha256 sha384 sha512 sha3_224 sha3_256 sha3_384 sha3_512 blake2b blake2s new "
        "algorithms_available algorithms_guaranteed pbkdf2_hmac file_digest".split()
    ),
    "difflib": frozenset(
        "SequenceMatcher Differ HtmlDiff unified_diff context_diff ndiff get_close_matches IS_LINE_JUNK "
        "IS_CHARACTER_JUNK diff_bytes restore".split()
    ),
    "pathlib": frozenset("Path PurePath PurePosixPath PureWindowsPath PosixPath WindowsPath".split()),
    "sqlite3": frozenset(
        "connect Error DatabaseError OperationalError IntegrityError ProgrammingError InterfaceError DataError "
        "NotSupportedError InternalError Row PARSE_DECLTYPES PARSE_COLNAMES version sqlite_version "
        "register_adapter register_converter complete_statement".split()
    ),
    "hermes_tools": frozenset(
        "web_search web_extract read_file write_file patch terminal json_parse shell_quote retry".split()
    ),
}
_SAFE_MODULES = frozenset(_MODULE_ATTRS)
_HERMES_PURE_HELPERS = frozenset({"json_parse", "shell_quote", "retry"})

_SAFE_BUILTINS = frozenset(
    """
    abs all any ascii bin bool bytearray bytes callable chr complex dict divmod enumerate filter float format
    frozenset hasattr hash hex id int isinstance issubclass iter len list map max min next object oct ord pow print
    range repr reversed round set slice sorted str sum tuple zip property staticmethod classmethod super
    NotImplemented Ellipsis
    BaseException Exception ValueError KeyError TypeError IndexError AttributeError RuntimeError StopIteration
    FileNotFoundError OSError IOError NotImplementedError ZeroDivisionError AssertionError PermissionError
    UnicodeDecodeError UnicodeEncodeError LookupError ArithmeticError OverflowError FileExistsError
    NotADirectoryError IsADirectoryError TimeoutError KeyboardInterrupt SystemExit
    """.split()
)
# Names that mean dynamic code / introspection, wherever they appear (called, aliased, passed, defaulted).
_BANNED_NAMES = frozenset(
    "exec eval compile __import__ globals vars locals getattr setattr delattr breakpoint type help builtins "
    "__builtins__ __loader__ __spec__ __build_class__ memoryview".split()
)
_BANNED_ATTRS = frozenset(
    """
    load_extension enable_load_extension get_type_hints ForwardRef evaluate_forward_ref Formatter vformat get_field
    get_value import_module f_globals f_locals f_builtins f_back f_code gi_frame gi_code cr_frame cr_code ag_frame
    ag_code tb_frame tb_next co_code func_globals
    """.split()
)
_DUNDER_STRING = re.compile(r"__[A-Za-z_][A-Za-z0-9_]*__")

# Methods that are harmless on every object a SAFE program can hold (str/list/dict/set/regex/datetime/hash/csv/file/
# sqlite handles/pure Path operations). Everything else called on an object is unrecognised -> REVIEW.
_SAFE_METHODS = frozenset(
    """
    capitalize casefold center count encode endswith expandtabs find format format_map index isalnum isalpha isascii
    isdecimal isdigit isidentifier islower isnumeric isprintable isspace istitle isupper join ljust lower lstrip
    partition removeprefix removesuffix replace rfind rindex rjust rpartition rsplit rstrip split splitlines
    startswith strip swapcase title translate upper zfill maketrans decode hex fromhex
    append extend insert pop remove sort reverse copy clear keys values items get setdefault update popitem fromkeys
    add discard union intersection difference symmetric_difference issubset issuperset isdisjoint appendleft popleft
    rotate most_common elements subtract total
    group groups groupdict start end span expand match search fullmatch findall finditer sub subn
    now today utcnow fromtimestamp utcfromtimestamp fromisoformat strptime strftime isoformat timestamp date time
    weekday isoweekday isocalendar timetuple astimezone utcoffset total_seconds combine toordinal fromordinal
    hexdigest digest
    writerow writerows writeheader
    read readline readlines write writelines close flush seek tell
    cursor fetchone fetchall fetchmany commit rollback
    home cwd joinpath resolve expanduser absolute with_name with_suffix with_stem relative_to is_relative_to
    as_posix exists is_file is_dir is_symlink is_absolute stat lstat iterdir glob rglob
    """.split()
)
# Path operations that touch file contents / the filesystem and therefore need a foldable receiver.
_PATH_READ = frozenset({"read_text", "read_bytes"})
_PATH_WRITE = frozenset({"write_text", "write_bytes", "mkdir", "touch", "chmod"})
_PATH_DELETE = frozenset({"unlink", "rmdir"})
_PATH_MOVE = frozenset({"rename", "replace"})
# Python 3.14 ``Path.copy`` / ``copy_into`` / ``move`` / ``move_into``.
_PATH_TRANSFER = frozenset({"copy", "copy_into", "move", "move_into"})
# Names that are in ``_SAFE_METHODS`` (str/list/dict/datetime methods) but are also path operations: using them as a
# value (``f = p.copy``) would hide the call from the path vetting.
_ALIAS_UNSAFE = frozenset({"copy", "replace"})
_PATH_LINK = frozenset({"symlink_to", "hardlink_to", "link_to"})
_SQL_METHODS = frozenset({"execute", "executemany", "executescript"})
# Effectful names must appear directly as the callee, where they are vetted; aliasing or passing them on
# (``f = p.write_text``, ``map(open, ...)``) would escape the vetting.
_EFFECT_ATTRS = (
    _PATH_READ | _PATH_WRITE | _PATH_DELETE | _PATH_MOVE | _PATH_LINK | _SQL_METHODS
    | frozenset({"open", "connect", "terminal", "read_file", "write_file", "patch", "search_files", "web_search",
                 "web_extract", "rmtree", "system", "popen", "remove"})
)
_EFFECT_ORIGINS = frozenset(
    {"sqlite3.connect"} | {f"hermes_tools.{n}" for n in _MODULE_ATTRS["hermes_tools"] - _HERMES_PURE_HELPERS}
)

_PATH_CLASSES = frozenset({"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"})
_DECORATOR_ORIGINS = frozenset(
    {"dataclasses.dataclass", "functools.lru_cache", "functools.cache", "functools.wraps",
     "functools.cached_property", "functools.total_ordering", "functools.singledispatch"}
)
_DECORATOR_BUILTINS = frozenset({"staticmethod", "classmethod", "property"})
_CLASS_BASE_ORIGINS = frozenset({"typing.NamedTuple", "typing.TypedDict", "typing.Generic", "typing.Protocol"})

_PROC_ORIGINS = re.compile(
    r"^(subprocess\.(run|call|check_call|check_output|Popen|getoutput|getstatusoutput)"
    r"|os\.(system|popen|exec\w*|spawn\w*|posix_spawn\w*|startfile)"
    r"|pty\.spawn|asyncio\.create_subprocess_(shell|exec)|commands\.getoutput)$"
)
_DELETE_ORIGINS = frozenset(
    {"shutil.rmtree", "os.remove", "os.unlink", "os.rmdir", "os.removedirs"}
)
_DECODE_NAMES = frozenset(
    "b64decode b32decode b16decode a85decode b85decode standard_b64decode urlsafe_b64decode decodebytes unhexlify "
    "a2b_base64 a2b_hex decompress rot13 decode".split()
)
_DECODE_MODULES = frozenset({"base64", "codecs", "binascii", "zlib", "marshal", "bz2", "lzma", "gzip"})
_SECRET_KEY = re.compile(r"(?i)(key|token|secret|passw|credential|auth|cookie|session)")

# Credential-ish paths (case-insensitive fnmatch; ``*`` crosses ``/``).
_CRED_GLOBS = (
    "**/.env*", "**/auth.json", "**/.ssh/**", "/proc/*/environ", "**/*credentials*", "**/*token*", "**/.netrc",
    "**/.aws/**", "**/.git-credentials", "**/id_rsa*", "**/id_ed25519*", "**/.npmrc", "**/.pypirc",
    "/etc/shadow*", "/etc/sudoers*",
    "**/.config/gh/**", "**/.kube/config", "**/.docker/config.json", "**/.gnupg/**",
)
# Hermes's own config and source: a script must never rewrite these on its own.
_SELF_MODIFY_GLOBS = (
    "**/.hermes/config.yaml", "**/.hermes/profiles/*/config.yaml", "**/.hermes/hermes-agent/**",
    "**/.hermes/hermes-agent",
)

_PARENT = "_scan_parent"

_ALLOWED_NODES: frozenset[type] = frozenset(
    t for t in (getattr(ast, n, None) for n in """
    Module Expr Assign AugAssign AnnAssign If For While Break Continue Pass Return FunctionDef Lambda With withitem
    Try ExceptHandler Raise Assert Delete ListComp SetComp DictComp GeneratorExp comprehension BoolOp BinOp UnaryOp
    Compare IfExp Call Attribute Subscript Slice Name Constant JoinedStr FormattedValue List Tuple Set Dict Starred
    NamedExpr Load Store Del And Or Add Sub Mult MatMult Div Mod Pow LShift RShift BitOr BitXor BitAnd FloorDiv
    Invert Not UAdd USub Eq NotEq Lt LtE Gt GtE Is IsNot In NotIn arguments arg keyword Import ImportFrom alias
    Global Nonlocal ClassDef Yield YieldFrom Index ExtSlice
    """.split()) if t is not None
)


def _fnmatch_any(text: str, globs: Sequence[str]) -> str | None:
    lowered = text.lower()
    for pattern in globs:
        low = pattern.lower()
        if fnmatch.fnmatchcase(lowered, low) or fnmatch.fnmatchcase(lowered + "/", low):
            return pattern
        # ``**/x`` must also match a bare relative ``x`` (``.env`` read from the cwd).
        if low.startswith("**/") and fnmatch.fnmatchcase("/" + lowered.lstrip("/"), low):
            return pattern
    return None


def _is_cred_path(path: str) -> bool:
    return _fnmatch_any(path, _CRED_GLOBS) is not None


_SQL_STR = re.compile(r"'(?:[^']|'')*'")
_SQL_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|attach|detach|drop|create|alter|vacuum|reindex|pragma|load_extension|analyze"
    r"|replace\s+into)\b",
    re.IGNORECASE,
)
_SQL_PRAGMA = re.compile(r"^pragma\s+(?P<name>[a-z_]+)\s*(?:\((?P<arg>[^=;()]*)\))?\s*;?\s*$", re.IGNORECASE)
# ``PRAGMA name(value)`` WRITES (``schema_version(1)``, ``journal_mode(OFF)`` ...), so only pragmas that merely look
# something up may take an argument ...
_PRAGMA_READ_WITH_ARG = frozenset(
    "table_info table_xinfo table_list index_list index_info index_xinfo foreign_key_list foreign_key_check "
    "integrity_check quick_check".split()
)
# ... and without one only pragmas that report a setting; ``wal_checkpoint``, ``optimize``, ``incremental_vacuum``
# and ``shrink_memory`` do work on the database even when bare.
_PRAGMA_READ_BARE = _PRAGMA_READ_WITH_ARG | frozenset(
    "user_version schema_version application_id journal_mode page_count page_size freelist_count encoding "
    "foreign_keys busy_timeout database_list collation_list compile_options function_list module_list pragma_list "
    "data_version max_page_count locking_mode synchronous temp_store cache_size auto_vacuum secure_delete "
    "wal_autocheckpoint mmap_size query_only recursive_triggers read_uncommitted".split()
)


def _sql_is_read(sql: str) -> bool:
    """True only for one read-only statement: SELECT / WITH / EXPLAIN without write keywords, or a read-only PRAGMA."""
    text = sql.strip()
    if re.match(r"(?i)^pragma\b", text):
        match = _SQL_PRAGMA.match(text)
        if match is None:
            return False
        name = match.group("name").lower()
        return name in (_PRAGMA_READ_WITH_ARG if match.group("arg") is not None else _PRAGMA_READ_BARE)
    if not re.match(r"(?is)^(select|with|explain)\b", text):
        return False
    body = _SQL_STR.sub("''", text).rstrip(";").strip()
    if ";" in body:
        return False
    return not _SQL_WRITE_WORDS.search(body)


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()[:16]


class _TildePath(str):
    """A path starting with ``~``: ``alt`` is the literal (cwd-relative) reading of the same text."""

    alt: str | None = None


class _Scanner(ast.NodeVisitor):
    def __init__(self, code: str, tree: ast.AST, *, write_roots: Sequence, cwd, deny_globs: Sequence[str],
                 hermes_home) -> None:
        self.code = code
        self.tree = tree
        self.cwd = os.path.realpath(os.fspath(cwd)) if cwd else os.getcwd()
        self.roots = [
            r for r in (os.path.realpath(os.fspath(p)) for p in write_roots if p) if r not in ("/", "")
        ]
        self.deny_globs = [g.strip() for g in deny_globs if isinstance(g, str) and g.strip()]
        home = os.path.expanduser("~")
        hermes = os.fspath(hermes_home) if hermes_home else os.path.join(home, ".hermes")
        self.self_modify_globs = _SELF_MODIFY_GLOBS + (
            os.path.join(hermes, "config.yaml"), os.path.join(hermes, "hermes-agent", "**"),
        )
        self.home = home
        # Reading these dot-directories under $HOME is routine for Hermes itself; any other dot entry is REVIEW.
        self.dot_exempt = [os.path.realpath(os.path.join(home, ".hermes")), os.path.realpath(hermes)]
        self.features: set[str] = set()
        self.review: list[str] = []
        self.high: list[str] = []
        self.imports: dict[str, str] = {}
        self.module_aliases: set[str] = set()
        # Methods the program itself defines, per class (their bodies are scanned too). A call is only trusted when the
        # receiver is provably an instance of such a class, never merely because some class has a method of that name.
        self.class_methods: dict[str, set[str]] = {}
        self.class_bases: dict[str, list[str]] = {}
        self.ambiguous: set[str] = set()
        self.bindings: dict[str, list[str]] = {}
        self.const_values: dict[str, ast.AST] = {}
        self._folding: set[str] = set()
        self.db_write_lines: list[int] = []
        self.connects: list[tuple[str | None, bool, bool, int]] = []  # (path, read_only, memory, line)
        self.saw_decode = False
        self.saw_exec = False

    # -- reporting ---------------------------------------------------------------------------------------------
    @staticmethod
    def _line(node) -> int:
        return getattr(node, "lineno", 0) or 0

    def rev(self, node, feature: str, why: str) -> None:
        self.features.add(feature)
        if len(self.review) < _MAX_REASONS:
            self.review.append(f"{why} at line {self._line(node)}")

    def hi(self, node, feature: str, why: str) -> None:
        self.features.add(feature)
        if len(self.high) < _MAX_REASONS:
            self.high.append(f"{why} at line {self._line(node)}")

    # -- pre-pass ----------------------------------------------------------------------------------------------
    def prepare(self) -> None:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, _PARENT, parent)
        bindings: dict[str, list[str]] = {}
        origins: dict[str, set[str]] = {}

        def bind(name: str, kind: str) -> None:
            bindings.setdefault(name, []).append(kind)

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                bind(node.name, "def")
                owner = getattr(node, _PARENT, None)
                if isinstance(owner, ast.ClassDef) and node.name not in _EFFECT_ATTRS:
                    self.class_methods.setdefault(owner.name, set()).add(node.name)
                for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                    bind(a.arg, "other")
                for a in (node.args.vararg, node.args.kwarg):
                    if a is not None:
                        bind(a.arg, "other")
            elif isinstance(node, ast.ClassDef):
                bind(node.name, "class")
                self.class_methods.setdefault(node.name, set())
                self.class_bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
            elif isinstance(node, ast.Lambda):
                for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                    bind(a.arg, "other")
                for a in (node.args.vararg, node.args.kwarg):
                    if a is not None:
                        bind(a.arg, "other")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    bind(bound, "import")
                    self.module_aliases.add(bound)
                    origins.setdefault(bound, set()).add(alias.name if alias.asname else alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound = alias.asname or alias.name
                    bind(bound, "import")
                    origins.setdefault(bound, set()).add(f"{'.' * node.level}{node.module or ''}.{alias.name}")
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bind(node.name, "other")
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    bind(name, "other")
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                parent = getattr(node, _PARENT, None)
                top_assign = (
                    isinstance(parent, (ast.Assign, ast.AnnAssign))
                    and isinstance(getattr(parent, _PARENT, None), ast.Module)
                    and (len(parent.targets) == 1 if isinstance(parent, ast.Assign) else True)
                    and getattr(parent, "value", None) is not None
                    and (parent.targets[0] if isinstance(parent, ast.Assign) else parent.target) is node
                )
                bind(node.id, "assign-top" if top_assign else "other")
                if top_assign:
                    self.const_values[node.id] = parent.value
        self.bindings = bindings
        for name, kinds in bindings.items():
            if "import" in kinds:
                if len(kinds) == 1 or (set(kinds) == {"import"} and len(origins[name]) == 1):
                    self.imports[name] = next(iter(origins[name]))
                else:
                    self.ambiguous.add(name)
            if len(origins.get(name, ())) > 1:
                self.imports.pop(name, None)
                self.ambiguous.add(name)

    def origin(self, node) -> str | None:
        if isinstance(node, ast.Name):
            return self.imports.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.origin(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    # -- constant folding --------------------------------------------------------------------------------------
    def fold(self, node):
        """``str`` / ``PurePosixPath`` value of an expression built only from literals, or None."""
        try:
            return self._fold(node, 0)
        except Exception:
            return None

    def _fold(self, node, depth: int):
        if depth > 40:
            return None
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.JoinedStr):
            out = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    out.append(part.value)
                elif isinstance(part, ast.FormattedValue) and part.conversion in (-1, None) and part.format_spec is None:
                    value = self._fold(part.value, depth + 1)
                    if value is None:
                        return None
                    out.append(str(value))
                else:
                    return None
            return "".join(out)
        if isinstance(node, ast.BinOp):
            left, right = self._fold(node.left, depth + 1), self._fold(node.right, depth + 1)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(node.op, ast.Div) and isinstance(left, PurePosixPath):
                return left / right
            return None
        if isinstance(node, ast.Name):
            if node.id in self._folding or node.id in self.ambiguous:
                return None
            kinds = self.bindings.get(node.id, [])
            value = self.const_values.get(node.id)
            if kinds != ["assign-top"] or value is None:
                return None
            self._folding.add(node.id)
            try:
                return self._fold(value, depth + 1)
            finally:
                self._folding.discard(node.id)
        if isinstance(node, ast.Attribute):
            base = self._fold(node.value, depth + 1)
            if isinstance(base, PurePosixPath):
                if node.attr == "parent":
                    return base.parent
                if node.attr in ("name", "stem", "suffix"):
                    return getattr(base, node.attr)
            return None
        if isinstance(node, ast.Subscript):
            # Path(...).parents[n]
            if isinstance(node.value, ast.Attribute) and node.value.attr == "parents" \
                    and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                base = self._fold(node.value.value, depth + 1)
                if isinstance(base, PurePosixPath):
                    try:
                        return base.parents[node.slice.value]
                    except IndexError:
                        return None
            return None
        if isinstance(node, ast.Call):
            return self._fold_call(node, depth)
        return None

    def _fold_call(self, node: ast.Call, depth: int):
        func = node.func
        if node.keywords:
            return None
        origin = self.origin(func)
        if origin in _PATH_CLASSES:
            parts = []
            for arg in node.args:
                value = self._fold(arg, depth + 1)
                if value is None:
                    return None
                parts.append(str(value))
            return PurePosixPath(*parts) if parts else PurePosixPath(".")
        if origin in ("pathlib.Path.home",) and not node.args:
            return PurePosixPath(self.home)
        if origin in ("pathlib.Path.cwd",) and not node.args:
            return PurePosixPath(self.cwd)
        if isinstance(func, ast.Name) and func.id == "str" and func.id not in self.bindings and len(node.args) == 1:
            value = self._fold(node.args[0], depth + 1)
            return None if value is None else str(value)
        if isinstance(func, ast.Attribute):
            base = self._fold(func.value, depth + 1)
            if isinstance(base, PurePosixPath):
                if func.attr == "expanduser" and not node.args:
                    text = str(base)
                    if text == "~" or text.startswith("~/"):
                        return PurePosixPath(self.home + text[1:])
                    return base
                if func.attr in ("resolve", "absolute") and not node.args:
                    return PurePosixPath(os.path.realpath(self._abs(base)))
                if func.attr == "as_posix" and not node.args:
                    return str(base)
                if func.attr == "joinpath":
                    parts = [self._fold(a, depth + 1) for a in node.args]
                    return None if any(p is None for p in parts) else base.joinpath(*map(str, parts))
                if func.attr in ("with_name", "with_suffix", "with_stem") and len(node.args) == 1:
                    arg = self._fold(node.args[0], depth + 1)
                    if isinstance(arg, str):
                        try:
                            return getattr(base, func.attr)(arg)
                        except ValueError:
                            return None
        return None

    def _abs(self, value) -> str:
        text = str(value)
        if not os.path.isabs(text):
            text = os.path.join(self.cwd, text)
        return os.path.normpath(text)

    def fold_path(self, node) -> str | None:
        value = self.fold(node) if node is not None else None
        if value is None or (isinstance(value, str) and not value):
            return None
        text = str(value)
        if text == "~" or text.startswith("~/"):
            # Python's open() takes "~" literally but Hermes's file tools expand it: do not guess which one runs.
            # Both readings are checked (see _file_op); the expanded one is the primary.
            primary = _TildePath(self._abs(self.home + text[1:]))
            primary.alt = self._abs(text)
            return primary
        return self._abs(value)

    # -- file operations ---------------------------------------------------------------------------------------
    def _under_roots(self, real: str) -> bool:
        return any(real == r or real.startswith(r + os.sep) for r in self.roots)

    @staticmethod
    def _has_git_component(path: str) -> bool:
        return any(part.lower() == ".git" for part in path.split(os.sep))

    def _in_roots(self, real: str) -> bool:
        """Inside a write root, but never in ``.git``: the next ``git status`` would run what a script put there
        (hooks, ``core.fsmonitor``, ``core.sshCommand`` ...), so ``.git`` is not part of any root."""
        for root in self.roots:
            if real == root or real.startswith(root + os.sep):
                if not self._has_git_component(real[len(root):]):
                    return True
        return False

    def _root_write_ok(self, path: str) -> bool:
        """``path`` (as written) and where it really lands are both inside a root and neither is in ``.git``."""
        return self._in_roots(os.path.realpath(path)) and not self._has_git_component(path)

    def _home_dot_entry(self, path: str) -> bool:
        """A dotfile / dot-directory directly under $HOME (``~/.bashrc``, ``~/.config/x``) that is neither Hermes's own
        nor inside a write root."""
        for home in {self.home, os.path.realpath(self.home)}:
            if path.startswith(home + os.sep):
                first = path[len(home) + 1:].split(os.sep, 1)[0]
                if first.startswith(".") and first not in (".", ".."):
                    if any(path == e or path.startswith(e + os.sep) for e in self.dot_exempt):
                        return False
                    return not self._under_roots(os.path.realpath(path))
        return False

    def _file_op(self, node, path: str | None, kind: str, what: str) -> None:
        """``kind`` is read | write | delete; ``path`` an absolute, normalised path or None (not foldable)."""
        alt = getattr(path, "alt", None)
        if alt is not None:
            self._file_op(node, str(path), kind, what)
            self._file_op(node, alt, kind, what)
            return
        if path is None:
            if kind == "delete":
                self.hi(node, "del:dynamic", f"{what}: delete target is not a constant path")
            else:
                self.rev(node, f"{kind}:dynamic", f"{what}: {kind} path is not a constant")
            return
        real = os.path.realpath(path)
        candidates = {path, real}
        denied = next((p for p in candidates if self.deny_globs and _fnmatch_any(p, self.deny_globs)), None)
        if denied:
            self.hi(node, {"read": "read:cred", "write": "write:outside", "delete": "del:outside"}[kind],
                    f"{what}: path matches an approvals.deny rule")
        if kind == "read":
            if any(_is_cred_path(p) for p in candidates):
                self.hi(node, "read:cred", f"{what}: reads a credential-like path")
            elif any(self._home_dot_entry(p) for p in candidates):
                self.features.add("read:fs")
                self.rev(node, "read:homedot", f"{what}: reads a dotfile or dot-directory under $HOME")
            else:
                self.features.add("read:fs")
            return
        if any(_fnmatch_any(p, self.self_modify_globs) for p in candidates):
            self.hi(node, f"{kind}:outside", f"{what}: modifies Hermes's own config or source")
            return
        in_roots = self._in_roots(real) and not self._has_git_component(path)
        if kind == "delete":
            if in_roots:
                self.features.add("del:roots")
            else:
                self.hi(node, "del:outside", f"{what}: deletes outside the write roots")
            return
        if in_roots:
            self.features.add("write:roots")
        else:
            self.rev(node, "write:outside", f"{what}: writes outside the write roots")

    # -- visitor core ------------------------------------------------------------------------------------------
    def visit(self, node):
        if type(node) not in _ALLOWED_NODES:
            self.rev(node, f"syntax:{type(node).__name__}", f"unrecognised syntax {type(node).__name__}")
            return self.generic_visit(node)
        return super().visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and node.value != "__main__" and _DUNDER_STRING.search(node.value):
            self.rev(node, "obf:dynamic", "string literal names a dunder")

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            ok = alias.name in _SAFE_MODULES or alias.name == "collections.abc"
            if not ok:
                self.rev(node, f"import:{alias.name}", f"import of unvetted module {alias.name}")
            if not alias.name.isascii():
                self.rev(node, "ident:non-ascii", "non-ASCII module name")
            if top in _DECODE_MODULES:
                self.saw_decode = True

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if node.level or not module:
            self.rev(node, "import:relative", "relative import")
            return
        if module in _DECODE_MODULES or module.split(".")[0] in _DECODE_MODULES:
            self.saw_decode = True
        if module not in _SAFE_MODULES and module != "collections.abc":
            self.rev(node, f"import:{module}", f"import from unvetted module {module}")
            return
        allowed = _MODULE_ATTRS.get(module, _ANY)
        for alias in node.names:
            name = alias.name
            if name == "*":
                self.rev(node, f"import:{module}.*", "star import")
            elif name in _BANNED_ATTRS or name in _BANNED_NAMES:
                self.rev(node, "obf:dynamic", f"import of {module}.{name}")
            elif allowed is not None and name not in allowed:
                self.rev(node, f"import:{module}.{name}", f"import of unvetted {module}.{name}")
            elif allowed is None and name.startswith("_"):
                self.rev(node, f"import:{module}.{name}", f"import of private {module}.{name}")

    def visit_Name(self, node: ast.Name):
        name = node.id
        if not name.isascii():
            self.rev(node, "ident:non-ascii", f"non-ASCII identifier {name!r}")
        if not isinstance(node.ctx, ast.Load):
            return
        parent = getattr(node, _PARENT, None)
        is_callee = isinstance(parent, ast.Call) and parent.func is node
        if name in ("exec", "eval", "compile", "__import__"):
            self.saw_exec = True
        if name in _BANNED_NAMES:
            self.rev(node, "obf:dynamic", f"use of {name}")
            return
        if name.startswith("__") and name.endswith("__"):
            if name != "__name__":
                self.rev(node, "obf:dynamic", f"use of dunder name {name}")
            return
        if name in self.ambiguous:
            self.rev(node, f"name:ambiguous:{name}", f"name {name} is bound more than once incl. an import")
            return
        if name in self.imports:
            origin = self.imports[name]
            as_module_base = isinstance(parent, ast.Attribute) and parent.value is node
            if name in self.module_aliases and not as_module_base:
                self.rev(node, f"alias:module:{origin}", f"module {origin} used as a value")
            if origin in _EFFECT_ORIGINS and not is_callee:
                self.rev(node, f"alias:{origin}", f"{origin} used as a value")
            self._env_access(node, origin)
            return
        if name in self.bindings:
            if name == "open":
                self.rev(node, "alias:open", "open is rebound in the program")
            return
        if name == "open":
            if not is_callee:
                self.rev(node, "alias:open", "open used as a value")
            return
        if name in _SAFE_BUILTINS:
            return
        self.rev(node, f"name:unknown:{name}", f"unrecognised name {name}")

    def visit_Attribute(self, node: ast.Attribute):
        attr = node.attr
        parent = getattr(node, _PARENT, None)
        if not attr.isascii():
            self.rev(node, "ident:non-ascii", f"non-ASCII attribute {attr!r}")
        if attr.startswith("__") and attr.endswith("__"):
            self.rev(node, "obf:dynamic", f"dunder attribute {attr}")
        elif attr in _BANNED_ATTRS:
            self.rev(node, "obf:dynamic", f"introspection attribute {attr}")
        if attr in _DECODE_NAMES:
            self.saw_decode = True
        if isinstance(node.value, ast.Name) and node.value.id in _DECODE_MODULES:
            self.saw_decode = True
        base_origin = self.origin(node.value) if isinstance(node.value, ast.Name) else None
        if base_origin in _MODULE_ATTRS:
            allowed = _MODULE_ATTRS[base_origin]
            if allowed is not None and attr not in allowed:
                self.rev(node, f"attr:{base_origin}.{attr}", f"unvetted {base_origin}.{attr}")
            elif allowed is None and attr.startswith("_"):
                self.rev(node, f"attr:{base_origin}.{attr}", f"private {base_origin}.{attr}")
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in self.imports:
                self.rev(node, "patch:module", "assignment to an attribute of an imported object")
        is_callee = isinstance(parent, ast.Call) and parent.func is node
        if (attr in _EFFECT_ATTRS or attr in _PATH_TRANSFER) and not is_callee \
                and (attr not in _SAFE_METHODS or attr in _ALIAS_UNSAFE):
            self.rev(node, f"alias:{attr}", f"{attr} used as a value")
        origin = self.origin(node)
        if origin:
            self._env_access(node, origin)
            if origin in _EFFECT_ORIGINS and not is_callee:
                self.rev(node, f"alias:{origin}", f"{origin} used as a value")
        elif attr in ("environ", "environb"):
            self._env_access(node, f"?.{attr}")
        self.generic_visit(node)

    def _env_access(self, node, origin: str) -> None:
        """HIGH for dumping ``os.environ`` or reading a secret-looking variable; ``env:key`` for other literal keys."""
        is_getenv = origin.endswith(".getenv") or origin.endswith(".getenvb")
        if not (is_getenv or origin.endswith(".environ") or origin.endswith(".environb")):
            return
        parent = getattr(node, _PARENT, None)
        key = None
        if is_getenv:
            if not (isinstance(parent, ast.Call) and parent.func is node):
                return
            if parent.args:
                key = self._fold(parent.args[0], 0)
        elif isinstance(parent, ast.Subscript) and parent.value is node:
            key = self._fold(parent.slice, 0)
        elif isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = getattr(parent, _PARENT, None)
            if isinstance(call, ast.Call) and call.func is parent and call.args:
                key = self._fold(call.args[0], 0)
        elif isinstance(parent, ast.Compare) and node in parent.comparators:
            self.features.add("env:key")
            return
        if isinstance(key, str):
            if _SECRET_KEY.search(key):
                self.hi(node, "env:dump", f"reads secret-looking environment variable {key!r}")
            else:
                self.features.add("env:key")
        elif is_getenv:
            self.rev(node, "env:key", "getenv with a dynamic key")
        else:
            self.hi(node, "env:dump", "scans or dumps os.environ")

    # -- definitions -------------------------------------------------------------------------------------------
    def _decorators_ok(self, node) -> None:
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            origin = self.origin(target)
            plain = isinstance(target, ast.Name) and target.id in _DECORATOR_BUILTINS \
                and target.id not in self.bindings
            if not (origin in _DECORATOR_ORIGINS or plain):
                self.rev(dec, "syntax:decorator", "unrecognised decorator")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._decorators_ok(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self._decorators_ok(node)
        if node.keywords:
            self.rev(node, "syntax:metaclass", "class keyword arguments (metaclass)")
        for base in node.bases:
            origin = self.origin(base)
            plain = isinstance(base, ast.Name) and base.id not in self.bindings and base.id in _SAFE_BUILTINS
            local = isinstance(base, ast.Name) and self.bindings.get(base.id) == ["class"]
            if not (origin in _CLASS_BASE_ORIGINS or plain or local):
                self.rev(base, "syntax:class-base", "unrecognised base class")
        self.generic_visit(node)

    # -- calls -------------------------------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        func = node.func
        origin = self.origin(func)
        self._high_checks(node, origin)
        if isinstance(func, ast.Name):
            self._call_name(node, func, origin)
        elif isinstance(func, ast.Attribute):
            self._call_attr(node, func, origin)
        else:
            self.rev(node, "call:computed", "call of a computed callable")
        self.generic_visit(node)

    def _kwargs(self, node: ast.Call) -> dict | None:
        out: dict = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            out[kw.arg] = kw.value
        return out

    def _call_name(self, node: ast.Call, func: ast.Name, origin: str | None) -> None:
        name = func.id
        if origin is not None:
            self._call_origin(node, origin)
            return
        if name in self.bindings:
            if name == "open":
                return  # reported by visit_Name
            return  # local def/class/variable: its body is scanned; effectful callables are caught where they are named
        if name == "open":
            self._open_call(node, node.args, self._kwargs(node), what="open()")

    def _call_attr(self, node: ast.Call, func: ast.Attribute, origin: str | None) -> None:
        if origin is not None:
            self._call_origin(node, origin)
            return
        method = func.attr
        receiver = self.fold(func.value)
        is_path = isinstance(receiver, PurePosixPath)
        if method in _PATH_READ | _PATH_WRITE | _PATH_DELETE | _PATH_LINK or method == "open":
            self._path_method(node, func, method)
        elif method in _PATH_MOVE:
            if self._is_path_move(node, method, receiver):
                self._path_method(node, func, method)
            # str.replace(old, new) / datetime.replace(**kw) fall through as ordinary methods
        elif method in _PATH_TRANSFER and (is_path or (method == "copy" and (node.args or node.keywords))):
            if is_path:
                self._path_method(node, func, method)
            else:  # list.copy() / dict.copy() take no argument; with one this is Path.copy on a receiver we cannot see
                self.rev(node, "write:dynamic", f".{method}() with arguments on a receiver that is not a known Path")
        elif method in _SQL_METHODS:
            self._sql_call(node, method)
        elif method in _SAFE_METHODS or self._is_local_method_call(func):
            return
        else:
            self.rev(node, f"call:method:{method}", f"unrecognised method call .{method}()")

    @staticmethod
    def _is_path_move(node: ast.Call, method: str, receiver) -> bool:
        """``Path.rename(target)`` / ``Path.replace(target)`` versus ``str.replace(old, new)`` /
        ``datetime.replace(hour=0)``. Anything that can be the path operation is treated as one, keyword form included."""
        if isinstance(receiver, PurePosixPath) or method == "rename":
            return True
        if isinstance(receiver, str):
            return False
        names = {kw.arg for kw in node.keywords}
        if None in names or "target" in names:
            return True
        return len(node.args) == 1 and not names

    def _class_has_method(self, cls: str, method: str, _seen: frozenset = frozenset()) -> bool:
        if cls in _seen or self.bindings.get(cls) != ["class"]:
            return False
        if method in self.class_methods.get(cls, ()):
            return True
        return any(self._class_has_method(b, method, _seen | {cls}) for b in self.class_bases.get(cls, ()))

    def _instance_class(self, node) -> str | None:
        """Name of the program's own class when ``node`` is ``Cls(...)`` or a top-level ``x = Cls(...)`` variable."""
        if isinstance(node, ast.Name) and self.bindings.get(node.id) == ["assign-top"]:
            node = self.const_values.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and self.bindings.get(node.func.id) == ["class"]:
            return node.func.id
        return None

    def _is_local_method_call(self, func: ast.Attribute) -> bool:
        """``self.m()`` inside a method, or ``m()`` on a provable instance of a class the program defines itself and
        which defines ``m`` (its body is scanned like any other). A bare method NAME proves nothing about the receiver:
        ``mem.backup(...)`` is sqlite's even when some class of the program also defines ``backup``."""
        method, receiver = func.attr, func.value
        cls = self._instance_class(receiver)
        if cls is not None:
            return self._class_has_method(cls, method)
        if isinstance(receiver, ast.Name):
            fn = getattr(func, _PARENT, None)
            while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.Lambda, ast.Module)):
                fn = getattr(fn, _PARENT, None)
            if not isinstance(fn, ast.FunctionDef) or not fn.args.args or fn.args.args[0].arg != receiver.id:
                return False
            owner = getattr(fn, _PARENT, None)
            if not isinstance(owner, ast.ClassDef):
                return False
            rebound = any(
                isinstance(n, ast.Name) and n.id == receiver.id and isinstance(n.ctx, (ast.Store, ast.Del))
                for n in ast.walk(fn)
            )
            return not rebound and self._class_has_method(owner.name, method)
        return False

    def _call_origin(self, node: ast.Call, origin: str) -> None:
        parts = origin.split(".")
        root = parts[0]
        if origin == "sqlite3.connect":
            self._connect_call(node)
        elif root == "hermes_tools":
            self._tool_call(node, parts[1] if len(parts) > 1 else "")
        elif origin in _PATH_CLASSES or origin in ("pathlib.Path.home", "pathlib.Path.cwd"):
            return
        elif root not in _SAFE_MODULES:
            if root in _SAFE_BUILTINS:
                return
            self.features.add(f"call:{origin}")
            if len(self.review) < _MAX_REASONS:
                self.review.append(f"call to unvetted {origin} at line {self._line(node)}")
        elif len(parts) > 2 and parts[-1] not in _SAFE_METHODS and not parts[-1][:1].isupper():
            # <module>.<Class>.<method>(...) such as datetime.datetime.now(): the method name must be vetted too.
            self.rev(node, f"call:method:{parts[-1]}", f"unrecognised {origin}()")

    def _open_call(self, node, args, kwargs, *, what: str) -> None:
        if kwargs is None:
            self.rev(node, "write:dynamic", f"{what} with **kwargs")
            return
        if "opener" in kwargs:
            self.rev(node, "write:dynamic", f"{what} with a custom opener")
            return
        path_node = args[0] if args else kwargs.get("file")
        mode_node = args[1] if len(args) > 1 else kwargs.get("mode")
        mode = "r" if mode_node is None else self.fold(mode_node)
        if not isinstance(mode, str):
            self.rev(node, "write:dynamic", f"{what}: mode is not a constant")
            return
        path = self.fold_path(path_node)
        write = any(c in mode for c in "wax+")
        read = "r" in mode or "+" in mode
        if read:
            self._file_op(node, path, "read", what)
        if write:
            self._file_op(node, path, "write", what)

    def _path_method(self, node: ast.Call, func: ast.Attribute, method: str) -> None:
        what = f".{method}()"
        receiver = self.fold_path(func.value)
        kwargs = self._kwargs(node)
        if method in _PATH_LINK:
            self.rev(node, "write:symlink", f"{what}: creates a link")
            return
        if method == "open":
            if kwargs is None:
                self.rev(node, "write:dynamic", f"{what} with **kwargs")
                return
            mode_node = node.args[0] if node.args else kwargs.get("mode")
            mode = "r" if mode_node is None else self.fold(mode_node)
            if not isinstance(mode, str):
                self.rev(node, "write:dynamic", f"{what}: mode is not a constant")
                return
            if "r" in mode or "+" in mode:
                self._file_op(node, receiver, "read", what)
            if any(c in mode for c in "wax+"):
                self._file_op(node, receiver, "write", what)
        elif method in _PATH_READ:
            self._file_op(node, receiver, "read", what)
        elif method in _PATH_DELETE:
            self._file_op(node, receiver, "delete", what)
        elif method in _PATH_MOVE or method in _PATH_TRANSFER:
            moves = method in _PATH_MOVE or method.startswith("move")
            if moves:
                self._file_op(node, receiver, "delete", what)  # the source disappears
            else:
                self._file_op(node, receiver, "read", what)
            dest_node = None
            if kwargs is not None:  # ``**kwargs`` hides the destination: not foldable -> write:dynamic
                dest_node = node.args[0] if node.args else (kwargs.get("target") or kwargs.get("target_dir"))
            dest = self.fold_path(dest_node) if dest_node is not None else None
            if dest is not None and method.endswith("_into") and receiver is not None \
                    and getattr(dest, "alt", None) is None:
                dest = os.path.join(dest, os.path.basename(str(receiver)))  # lands as <dir>/<name>, name may be .git
            self._file_op(node, dest, "write", what)
        else:
            self._file_op(node, receiver, "write", what)

    def _sql_call(self, node: ast.Call, method: str) -> None:
        kwargs = self._kwargs(node) or {}
        sql_node = node.args[0] if node.args else kwargs.get("sql")
        sql = self.fold(sql_node) if sql_node is not None else None
        if not isinstance(sql, str):
            self.rev(node, "sql:dynamic", f".{method}(): SQL is not a constant")
            return
        if re.search(r"\b(attach|vacuum\s+into|load_extension)\b", sql, re.IGNORECASE):
            self.rev(node, "sql:attach", f".{method}(): ATTACH / VACUUM INTO / load_extension can reach any path")
            return
        if method == "executescript" or not _sql_is_read(sql):
            self.features.add("db:write")
            self.db_write_lines.append(self._line(node))
        else:
            self.features.add("db:read")

    def _connect_call(self, node: ast.Call) -> None:
        kwargs = self._kwargs(node)
        if kwargs is None:
            self.rev(node, "read:dynamic", "sqlite3.connect with **kwargs")
            return
        line = self._line(node)
        if any(isinstance(a, ast.Starred) for a in node.args):
            self.connects.append((None, False, False, line))
            self.rev(node, "read:dynamic", "sqlite3.connect with *args")
            return
        target = node.args[0] if node.args else kwargs.get("database")
        uri_node = node.args[7] if len(node.args) > 7 else kwargs.get("uri")   # connect(db, timeout, ..., uri)
        # True / False when the flag is a constant (SQLite uses its truthiness); None when it is not.
        uri: bool | None = False if uri_node is None else (bool(uri_node.value) if isinstance(uri_node, ast.Constant) else None)
        text = self.fold(target) if target is not None else None
        if not isinstance(text, str):
            self.connects.append((None, False, False, line))
            self.rev(node, "read:dynamic", "sqlite3.connect: database path is not a constant")
            return
        text = str(text)
        if text in ("", ":memory:"):
            self.connects.append((None, False, True, line))
            return
        read_only = False
        if text[:5].lower() == "file:" and uri is None:
            # Whether this is a URI or a literal file name decides which file SQLite opens: do not guess.
            self.connects.append((None, False, False, line))
            self.rev(node, "read:dynamic", "sqlite3.connect: file: target with a uri flag that is not a constant")
            return
        if text[:5].lower() == "file:" and uri:
            # SQLite percent-decodes the path ('%2E%2E' is '..') and takes mode= from the query string.
            rest = text[5:].partition("#")[0]
            body, _, query = rest.partition("?")
            if body.startswith("//"):
                authority, slash, tail = body[2:].partition("/")
                if authority not in ("", "localhost"):
                    self.connects.append((None, False, False, line))
                    self.rev(node, "read:dynamic", "sqlite3.connect: file: URI names a host")
                    return
                body = slash + tail
            text = unquote(body)
            if text != body:
                # SQLite percent-decodes the path, so '%2E%2E' is '..' and '%2F' is '/'. A body
                # that only looks in-root while being written is exactly the traversal case: do
                # not reason about it, ask the human.
                self.connects.append((None, False, False, line))
                self.rev(node, "read:dynamic", "sqlite3.connect: percent-encoded file: URI path")
                return
            if not text or "\x00" in text:
                self.connects.append((None, False, False, line))
                self.rev(node, "read:dynamic", "sqlite3.connect: file: URI path is empty or contains NUL")
                return
            modes = [v for k, v in parse_qsl(query, keep_blank_values=True) if k == "mode"]
            if modes and modes[-1] == "memory":
                self.connects.append((None, False, True, line))
                return
            read_only = bool(modes) and modes[-1] == "ro"
        path = self._abs(text)
        self.connects.append((path, read_only, False, line))
        self._file_op(node, path, "read", "sqlite3.connect")

    def _tool_call(self, node: ast.Call, tool: str) -> None:
        kwargs = self._kwargs(node)
        if kwargs is None:
            self.rev(node, f"tool:{tool}", f"hermes_tools.{tool} with **kwargs")
            return
        if tool in _HERMES_PURE_HELPERS:
            return
        if tool == "terminal":
            self.features.add("proc:gated")
            return
        if tool in ("web_search", "web_extract"):
            arg = node.args[0] if node.args else kwargs.get("query" if tool == "web_search" else "urls")
            values = []
            if isinstance(arg, (ast.List, ast.Tuple)):
                values = [self.fold(e) for e in arg.elts]
            elif arg is not None:
                values = [self.fold(arg)]
            if values and all(isinstance(v, str) for v in values):
                self.features.add("net:fixed")
            else:
                self.rev(node, "net:dynamic", f"{tool}: argument is not a constant (data could leave the host)")
            return
        if "cross_profile" in kwargs:
            self.rev(node, f"tool:{tool}", f"{tool} with cross_profile")
            return
        if tool in ("read_file", "write_file", "patch"):
            path_node = node.args[0] if node.args else kwargs.get("path")
            path = self.fold_path(path_node) if path_node is not None else None
            self._file_op(node, path, "read" if tool == "read_file" else "write", f"{tool}()")
            return
        self.rev(node, f"tool:{tool}", f"hermes_tools.{tool} is not vetted")

    # -- high-risk patterns (independent of whether the rest is recognised) ------------------------------------
    def _high_checks(self, node: ast.Call, origin: str | None) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("exec", "eval", "compile", "__import__"):
            self.saw_exec = True
        if origin and origin.split(".")[-1] in ("import_module", "exec_module"):
            self.saw_exec = True
        if self.deny_globs and len(self.code) <= _DENY_UNPARSE_LIMIT:
            try:
                text = ast.unparse(node)[:4000]
            except Exception:
                text = ""
            if text and _fnmatch_any(text, self.deny_globs):
                self.hi(node, "proc:direct", "call text matches an approvals.deny rule")
        if origin and _PROC_ORIGINS.match(origin):
            self.features.add("proc:direct")
            self.rev(node, "proc:direct", f"direct process execution via {origin}")
            command = self._command_text(node)
            if command is None:
                self.features.add("proc:nonconst")  # the command is not a constant: one grant would cover any of them
            elif self._dangerous_command(command):
                self.hi(node, "proc:direct", f"{origin} runs a command flagged as dangerous")
        elif origin in _DELETE_ORIGINS or (isinstance(func, ast.Attribute) and func.attr == "rmtree"):
            target = self.fold_path(node.args[0]) if node.args else None
            self._file_op(node, target, "delete", f"{origin or func.attr}()")

    def _command_text(self, node: ast.Call) -> str | None:
        if not node.args:
            return None
        first = node.args[0]
        if isinstance(first, (ast.List, ast.Tuple)):
            parts = [self.fold(e) for e in first.elts]
            return " ".join(map(str, parts)) if parts and all(isinstance(p, str) for p in parts) else None
        value = self.fold(first)
        return value if isinstance(value, str) else None

    def _dangerous_command(self, command: str) -> bool:
        if self.deny_globs and _fnmatch_any(command, self.deny_globs):
            return True
        try:
            from tools.approval_detection import detect_dangerous_command, detect_hardline_command
            return bool(detect_dangerous_command(command)[0] or detect_hardline_command(command)[0])
        except Exception:
            return True  # cannot tell -> treat as dangerous (human decides)

    # -- finish ------------------------------------------------------------------------------------------------
    def finish(self) -> None:
        if self.saw_decode and self.saw_exec:
            self.hi(self.tree, "obf:dynamic", "decoded data and dynamic execution in the same program")
        if self.db_write_lines:
            line = self.db_write_lines[0]
            ok = bool(self.connects) and all(
                memory or (path is not None and self._root_write_ok(path) and not read_only)
                for path, read_only, memory, _ in self.connects
            )
            if not ok:
                self.features.add("db:write")
                self.review.append(f"SQLite write whose database is not inside the write roots at line {line}")
            else:
                self.features.add("write:roots")
            for path, _, memory, _ in self.connects:
                if path and any(_fnmatch_any(p, self.self_modify_globs) for p in (path, os.path.realpath(path))):
                    self.features.add("write:outside")
                    self.high.append(f"SQLite write to Hermes's own config or source at line {line}")


def scan_python(code: str, *, write_roots: Sequence, cwd, deny_globs: Sequence[str] = (),
                hermes_home=None) -> CodeScan:
    """Classify ``code`` as SAFE / REVIEW / HIGH. Never raises; any internal failure is REVIEW."""
    digest = _digest(code)
    opaque = frozenset({f"opaque:{digest}"})
    n_bytes = len(code.encode("utf-8", "surrogatepass"))
    if n_bytes > _MAX_SCAN_BYTES:
        return CodeScan("review", opaque | {"size:large"}, (f"program is {n_bytes} bytes; not scanned",))
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return CodeScan("review", opaque | {"syntax:error"}, (f"cannot parse with this interpreter: {type(exc).__name__}",))
    try:
        scanner = _Scanner(code, tree, write_roots=write_roots, cwd=cwd, deny_globs=deny_globs,
                           hermes_home=hermes_home)
        scanner.prepare()
        scanner.visit(tree)
        scanner.finish()
    except Exception as exc:  # RecursionError on pathological nesting included
        logger.debug("code scan failed: %s: %s", type(exc).__name__, exc)
        return CodeScan("review", opaque | {"scan:error"}, (f"scan failed: {type(exc).__name__}",))
    features = set(scanner.features)
    reasons = list(scanner.high) + list(scanner.review)
    too_big = n_bytes > MAX_SAFE_BYTES or code.count("\n") + 1 > MAX_SAFE_LINES
    if too_big:
        features.add("size:large")
        reasons.append(f"program exceeds the SAFE size limit ({MAX_SAFE_LINES} lines / {MAX_SAFE_BYTES} bytes)")
    if scanner.high:
        tier: Tier = "high"
    elif scanner.review or too_big:
        tier = "review"
    else:
        tier = "safe"
    if tier != "safe" and too_big:
        features.add(f"opaque:{digest}")
    return CodeScan(tier, frozenset(features), tuple(reasons[:_MAX_REASONS]))
