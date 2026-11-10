"""Freeze worktree bytes without touching its index; import without checkout."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import uuid


def git(root, *args, input=None, env=None, check=True):
    result = subprocess.run(["git", "-C", str(root), *args], input=input, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode().strip()


def common_root(root):
    common = Path(git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = Path(root) / common
    return common.resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_tree(root, commit):
    entries = git(root, "ls-tree", "-rz", commit).split("\0")
    for entry in filter(None, entries):
        meta, name = entry.split("\t", 1)
        if (meta.startswith("160000 ") or name in {".remote.json", ".remote.env", ".remote.cargo.toml", "local.mk"}
                or name.split("/", 1)[0] in {".scratch", ".toolchain", "models", "target", "node_modules"}
                or name.startswith(("kernel/build/", "web/node_modules/"))):
            raise ValueError("candidate contains a private/build path or submodule: " + name)


@contextmanager
def freeze(root, base, revision=None):
    root = Path(root).resolve()
    ref = "refs/ignis-experiments/export/" + uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="ignis-experiment-") as temporary:
            temporary = Path(temporary)
            head = git(root, "rev-parse", "--verify", (revision or "HEAD") + "^{commit}")
            if revision:
                snapshot = head
            else:
                env = dict(os.environ, GIT_INDEX_FILE=str(temporary / "index"))
                # read-tree + add captures working files, including unstaged changes;
                # the user's actual index and branch are never written.
                git(root, "read-tree", head, env=env)
                git(root, "add", "--all", "--", ".", env=env)
                tree = git(root, "write-tree", env=env)
                env.update(GIT_AUTHOR_NAME="Ignis experiment", GIT_AUTHOR_EMAIL="experiment@localhost",
                           GIT_COMMITTER_NAME="Ignis experiment", GIT_COMMITTER_EMAIL="experiment@localhost",
                           GIT_AUTHOR_DATE="@0 +0000", GIT_COMMITTER_DATE="@0 +0000")
                snapshot = git(root, "commit-tree", tree, "-p", head, env=env,
                               input=b"Ignis frozen experiment source\n")
            check_tree(root, snapshot)
            tree = git(root, "rev-parse", snapshot + "^{tree}")
            git(root, "update-ref", ref, snapshot)
            exclusions = []
            if base and git(root, "rev-parse", "--verify", base + "^{commit}", check=False):
                exclusions = ["^" + base]
            # A bundle cannot be empty. The frozen commit normally supplies a new
            # object; an explicitly selected existing commit needs no transfer.
            bundle = temporary / "source.bundle"
            if git(root, "rev-list", snapshot, *exclusions):
                git(root, "bundle", "create", str(bundle), ref, *exclusions)
            source = dict(head=head, snapshot=snapshot, tree=tree, ref=ref,
                          dirty=tree != git(root, "rev-parse", head + "^{tree}"))
            yield source, bundle if bundle.exists() else None
    finally:
        git(root, "update-ref", "-d", ref, check=False)


def receive(root, source, bundle):
    if bundle and Path(bundle).exists():
        git(root, "bundle", "verify", str(bundle))
        ref = "refs/ignis-experiments/sources/" + source["snapshot"]
        git(root, "fetch", "--quiet", "--no-tags", "--no-write-fetch-head", str(bundle),
            source["ref"] + ":" + ref)
        if git(root, "rev-parse", ref) != source["snapshot"]:
            raise ValueError("bundle differs from frozen source")
    if git(root, "rev-parse", source["snapshot"] + "^{tree}") != source["tree"]:
        raise ValueError("candidate tree mismatch")
    check_tree(root, source["snapshot"])
