import type { ImageId, ImageStore } from "../store/imageStore.ts";
import { newKey } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import { makeThumb } from "../store/pictures.ts";
import { pdfText } from "./attach.ts";
import {
  type Author,
  authorLabel,
  bytesOf,
  type ChatAuthor,
  cleanName,
  type FileKind,
  freedBy,
  KIND_LABEL,
  kindOf,
  latest,
  present,
  type StoredBytes,
  uniqueName,
  USER,
  type Version,
  type Workspace,
  WORKSPACE_FORMAT,
  type WorkspaceFile,
  type WorkspaceId,
} from "./files.ts";

// The workspace module (spec #170): the only writer of workspaces, their
// files and their versions, in the page's store (ADR 0073) — IndexedDB, or
// the in-memory store without it. It has two entrances: the user's actions
// (the files a prompt attaches, and the panel's: its files, hand edits and
// restores) and the model's tool
// calls (tools.ts, through the local tools' router). Both come here, so every
// change is a version with its author, and every change is announced to the
// page through `subscribe`.
//
// A workspace is a document of `workspaces`; its files are documents of a
// collection of their own, so that reading one workspace reads none of the
// others. Writes run one after another, the page over: a version's number is
// read, counted up and written back, and two writes at once would both take
// the same one.
//
// The guard: on the model's side every chat, and every sub-agent of a chat,
// is a reader, and the module keeps for each reader and each file the version
// that reader last read or wrote. A reader's write over an existing file, its
// edit or its deletion is refused when the file is at another version now, so
// that no reader overwrites a change it has not seen: the user's, another
// chat's or another sub-agent's. The check runs in the queue of writes, with
// the change it allows. A chat's reader starts from what its history read
// (tools.ts, seenIn), which is kept with the chat; a sub-agent's starts empty
// and lasts its run.

export const WORKSPACES = "workspaces";

/** The collection holding a workspace's files, one document per file, keyed by the file's id. */
export const filesCollection = (workspace: WorkspaceId) => `workspace-files/${workspace}`;

/** A file at one of its versions, by the file's id, with its name then: one a read showed, or one a change made (`Written`). */
export type FileAt = { workspace: WorkspaceId; file: string; name: string; version: number };

/** A version a write or an addition made, as a reply's card names it. */
export type Written = FileAt;

/** A file the user brings: its text inline, or its bytes with the text derived from them. */
export type Incoming = { name: string; kind: FileKind; text?: string; blob?: Blob; derived?: string };

/**
 * On the model's side, who reads and writes the files: a chat, or a sub-agent
 * of a chat. Only the module makes readers, and only it keeps what each has
 * seen; a reader made anywhere else has seen nothing.
 */
export type Reader = { readonly author: ChatAuthor };

/** Who changes a file: the user, whom the guard never refuses, or a reader, under the guard. */
export type Changer = typeof USER | Reader;

/** A version of a file a reader has seen: it read it, or made it. */
export type Seen = { file: string; version: number };

/** What a chat's turn holds of the chat's workspace. */
export type ChatBinding = {
  /** The chat's key in the store: the chat as an author and a reader. */
  readonly key: string;
  /** The workspace the chat is bound to, if it is. */
  bound(): WorkspaceId | undefined;
  /**
   * The chat's workspace: the one it is bound to while the store holds it,
   * else a new one named after the chat, reserved once however many calls ask
   * at once, which the store keeps with its first file (`reserve`). A chat
   * whose workspace was deleted, in this tab or another, gets a new one so.
   */
  workspace(): Promise<WorkspaceId>;
  /** The chat's model as a reader: it has seen what its history read or made, and what it reads or makes in this turn. */
  readonly reader: Reader;
  /** A sub-agent of the chat, started by the `agent` call `agent.call`: a reader of its own, which has seen nothing yet. */
  agent(agent: { call: string; name: string }): Reader;
};

export type BindingOptions = {
  bound: () => WorkspaceId | undefined;
  /** The new workspace's name: the chat's title as it is now. */
  name: () => string;
  /**
   * The workspace born for the chat, once the store keeps it with its first
   * file: the chat records it, in place of `replaced`, the deleted one it
   * named, if any, and never changes it after.
   */
  onBorn: (id: WorkspaceId, replaced?: WorkspaceId) => void;
  /** The versions the chat's history read or made (tools.ts, seenIn): what its reader has seen when the turn starts. */
  seen?: Iterable<Seen>;
};

export type WriteOutcome = { ok: true; written: Written; version: Version; created: boolean; workspace: Workspace } | { ok: false; error: string };

/** What a change to one file did: the version it made, or why it made none. */
export type FileOutcome = { ok: true; written: Written } | { ok: false; error: string };

/**
 * How the text of a version's bytes is computed on its first read (#395): a
 * PDF's by pdf.js, an audio file's by the server's transcription route while
 * the models listing says a speech model can serve. Injected, so a test needs
 * neither a browser nor a network.
 */
export type Deriving = {
  /** A PDF's text, page by page. */
  pdfText: (data: ArrayBuffer) => Promise<string>;
  /** The speech model's availability as the models listing says it now: undefined when the server says nothing. */
  /** An audio file's transcript; rejects with the server's reason. */
};

/** The text a read gives of a version, none for a kind that has no text (a picture, any other file), or why it cannot be had now. */
export type Readable = { ok: true; text: string | undefined } | { ok: false; error: string };

export type Workspaces = {
  /** Every workspace, the last changed first. */
  list(): Promise<Workspace[]>;
  get(id: WorkspaceId): Promise<Workspace | null>;
  /** A workspace's files, deleted ones included, by name. */
  files(id: WorkspaceId): Promise<WorkspaceFile[]>;
  file(workspace: WorkspaceId, id: string): Promise<WorkspaceFile | null>;
  /** A version's content, as a download gives it: its text, or its bytes; null when the page holds no bytes of it. */
  content(version: Version): Promise<Blob | null>;
  create(name: string): Promise<Workspace>;
  /**
   * A new workspace that the store keeps only with its first file, written by
   * `add` or `writeText`: one whose first file the store refuses leaves no
   * workspace behind (spec #170, a workspace is born with its first file).
   * Until then `list` and `get` do not know it.
   */
  reserve(name: string): Workspace;
  rename(id: WorkspaceId, name: string): Promise<{ ok: true; workspace: Workspace } | { ok: false; error: string }>;
  /**
   * Deletes a workspace with its files and every version, and frees the bytes
   * nothing else holds: no other workspace, nor among the pictures `elsewhere`
   * names (the chats' and the image sessions'; null while not known, and then
   * no picture is freed). The chats that worked in it are the caller's.
   */
  remove(id: WorkspaceId, elsewhere: Iterable<ImageId> | null): Promise<{ files: number }>;
  /** The image store's pictures the workspaces hold, every version's: what deleting a chat or an image session keeps. */
  pictures(): Promise<Set<ImageId>>;
  /** A chat's hold on its workspace, for its turns. */
  chat(key: string, options: BindingOptions): ChatBinding;
  /** The file's newest version, as `reader` receives it: from then on the guard holds it as the version that reader read. */
  read(reader: Reader, file: WorkspaceFile): Version;
  /** Creates a text file, or replaces one whole with a new version. */
  writeText(workspace: WorkspaceId, name: string, text: string, by: Changer): Promise<WriteOutcome>;
  /** Replaces the one occurrence of `old` in the text file `file` (its id) with `replacement`, as a new version. */
  editText(workspace: WorkspaceId, file: string, old: string, replacement: string, by: Changer): Promise<WriteOutcome>;
  /**
   * New files, each under a name no other file of the workspace holds: the
   * user's, or a reader's (the pictures make_image makes), which has seen
   * the versions it adds.
   */
  add(workspace: WorkspaceId, files: Incoming[], by: Changer): Promise<Written[]>;
  /**
   * Deletes a file: a new version by `by` marking it deleted, the versions
   * before it kept in its history. The one way to delete, the user's and the
   * model's; a reader's deletion is under the guard.
   */
  deleteFile(workspace: WorkspaceId, file: string, by: Changer): Promise<FileOutcome>;
  /**
   * Gives a file another name, unique in its workspace, deleted files' names
   * included: their history keeps them. The content is not changed, so no
   * version is made; the file keeps its id and its kind.
   */
  renameFile(workspace: WorkspaceId, file: string, name: string): Promise<FileOutcome>;
  /**
   * The user's restore of version `n` of a file: a new newest version with
   * that version's content (its text, its bytes and the text derived from
   * them), authored by the user, the versions between kept. A deleted file is
   * restored the same way, from a version before its deletion.
   */
  restore(workspace: WorkspaceId, file: string, n: number): Promise<FileOutcome>;
  /**
   * The text the model reads of a version: a text file's own, or the text
   * derived from its bytes, a PDF's text or an audio file's transcript,
   * computed on the version's first read and kept with it, so every read
   * after it gives the same and asks nobody. Reads of one version at once
   * share one computation, stopped once every read waiting on it has stopped.
   */
  readable(file: WorkspaceFile, version: Version, signal?: AbortSignal): Promise<Readable>;
  /** Called after every change, with the workspace and the file it touched. */
  subscribe(listener: (change: { workspace: WorkspaceId; file?: string }) => void): () => void;
};

const isWorkspace = (value: unknown): value is Workspace =>
  typeof value === "object" && value !== null && (value as Workspace).format === WORKSPACE_FORMAT && typeof (value as Workspace).id === "string";

const isFile = (value: unknown): value is WorkspaceFile =>
  typeof value === "object" && value !== null && (value as WorkspaceFile).format === WORKSPACE_FORMAT && Array.isArray((value as WorkspaceFile).versions);

const isReader = (by: Changer): by is Reader => "author" in by;

const authorOf = (by: Changer): Author => (isReader(by) ? by.author : USER);

/** The file of the workspace named `name`: the one in the workspace, else a deleted one. */
const named = (files: WorkspaceFile[], name: string) =>
  files.find((f) => f.name === name && present(f)) ?? files.find((f) => f.name === name);

/** Where `old` occurs in `text`, overlapping occurrences included. */
function occurrences(text: string, old: string): number[] {
  const at: number[] = [];
  for (let i = text.indexOf(old); i !== -1; i = text.indexOf(old, i + 1)) at.push(i);
  return at;
}

const lineAt = (text: string, index: number) => text.slice(0, index).split("\n").length;

/** "3", "3 and 9", "3, 9 and 14". */
const listed = (items: number[]) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);

/** A picture's size in pixels, and a thumbnail to keep beside it; rejects when the browser cannot decode it. */
export type MeasurePicture = (blob: Blob) => Promise<{ width: number; height: number; thumb?: Blob }>;

export function createWorkspaces(
  store: ImageStore,
  options: { now?: () => number; newId?: () => string; measure?: MeasurePicture; derive?: Deriving } = {},
): Workspaces {
  const now = options.now ?? Date.now;
  const newId = options.newId ?? newKey;
  const measure = options.measure ?? ((blob: Blob) => makeThumb(blob));
  const derive = options.derive;
  // The text being computed for a version, by the file's id and the version's number.
  const deriving = new Map<string, { result: Promise<Readable>; stop: AbortController; waiting: number }>();
  const listeners = new Set<(change: { workspace: WorkspaceId; file?: string }) => void>();
  // The workspaces reserved and not kept yet, waiting for their first file, with what keeping one tells.
  const pending = new Map<WorkspaceId, { workspace: Workspace; onKept?: (id: WorkspaceId) => void }>();
  // The workspace reserved for each chat with none, by the chat's key: deleting a workspace forgets the ones reserved as it.
  const reservations = new Map<string, WorkspaceId>();
  // The guard's memory: for each reader, the newest version of each file (by its id) it read or made.
  const seen = new WeakMap<Reader, Map<string, number>>();
  let queue: Promise<unknown> = Promise.resolve();

  function newReader(author: ChatAuthor, from: Iterable<Seen> = []): Reader {
    const reader: Reader = Object.freeze({ author });
    seen.set(reader, new Map());
    for (const s of from) saw(reader, s.file, s.version);
    return reader;
  }

  /** `by` has seen version `n` of `file`: a version it read or made is never one it has not seen, whatever it saw since. */
  function saw(by: Changer, file: string, n: number) {
    const versions = isReader(by) ? seen.get(by) : undefined;
    versions?.set(file, Math.max(versions.get(file) ?? 0, n));
  }

  /** Why `by` may not change `file` as it is now, or null when it may: the user always may, a reader only over the version it last read or wrote. */
  function guard(by: Changer, file: WorkspaceFile): string | null {
    if (!isReader(by)) return null;
    const newest = latest(file);
    const had = seen.get(by)?.get(file.id);
    if (had === newest.n) return null;
    const who = authorLabel(newest.author, by.author);
    if (had === undefined) {
      return `${file.name} is a file you have not read: it is at version ${newest.n}, by ${who}. Read it with read_file before you change it.`;
    }
    if (newest.deleted) {
      return `${file.name} was deleted by ${who} since you last read it (version ${newest.n}). Read it again with read_file before you write it.`;
    }
    return `${file.name} changed since you last read it: it is at version ${newest.n}, by ${who}, and you last read version ${had}. Read it again with read_file before you change it.`;
  }

  /** `job` after every write before it; a failure fails that write alone. */
  function serial<T>(job: () => Promise<T>): Promise<T> {
    const next = queue.then(job);
    queue = next.catch(() => {});
    return next;
  }

  const notify = (change: { workspace: WorkspaceId; file?: string }) => {
    for (const listener of listeners) listener(change);
  };

  /**
   * An incoming file's bytes into the store: a picture into the image store,
   * with its size and a thumbnail, and as bytes the model cannot read when the
   * browser cannot decode it; anything else as a blob.
   */
  async function keep(item: Incoming): Promise<{ kind: FileKind; bytes?: StoredBytes }> {
    if (!item.blob) return { kind: item.kind };
    if (item.kind === "picture") {
      const measured = await measure(item.blob).catch(() => null);
      if (measured) {
        const { width, height, thumb } = measured;
        const image = await store.putImage(item.blob, { width, height, ...(thumb ? { thumb } : {}) });
        return { kind: "picture", bytes: { store: "image", id: image.id, type: image.type, size: image.bytes, width: image.width, height: image.height } };
      }
    }
    const blob = await store.putBlob(item.blob);
    return { kind: item.kind === "picture" ? "other" : item.kind, bytes: { store: "blob", id: blob.id, type: blob.type, size: blob.bytes } };
  }

  async function get(id: WorkspaceId): Promise<Workspace | null> {
    const value = await store.getDoc<unknown>(WORKSPACES, id);
    return isWorkspace(value) ? value : null;
  }

  async function files(id: WorkspaceId): Promise<WorkspaceFile[]> {
    const values = await store.listDocs<unknown>(filesCollection(id));
    return values.filter(isFile).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  }

  async function touch(id: WorkspaceId, at: number): Promise<Workspace> {
    const workspace = await get(id);
    if (!workspace) throw new Error(`no workspace has the id ${id}`);
    const changed = { ...workspace, updated: at };
    await store.putDoc(WORKSPACES, id, changed);
    return changed;
  }

  async function create(name: string): Promise<Workspace> {
    return serial(async () => {
      const at = now();
      const workspace: Workspace = { format: WORKSPACE_FORMAT, id: newId(), name: name.trim() || "Workspace", created: at, updated: at };
      await store.putDoc(WORKSPACES, workspace.id, workspace);
      notify({ workspace: workspace.id });
      return workspace;
    });
  }

  function reserve(name: string, onKept?: (id: WorkspaceId) => void): Workspace {
    const at = now();
    const workspace: Workspace = { format: WORKSPACE_FORMAT, id: newId(), name: name.trim() || "Workspace", created: at, updated: at };
    pending.set(workspace.id, { workspace, ...(onKept ? { onKept } : {}) });
    return workspace;
  }

  /** The workspace a new file goes into: the store's, or one reserved for its first file; null once it is deleted. */
  async function home(id: WorkspaceId): Promise<Workspace | null> {
    return pending.get(id)?.workspace ?? (await get(id));
  }

  /** The workspace changed at `at`: a reserved one is kept now, with the first file just written into it. */
  async function changed(id: WorkspaceId, at: number): Promise<Workspace> {
    const reserved = pending.get(id);
    if (!reserved) return touch(id, at);
    const workspace = { ...reserved.workspace, updated: at };
    await store.putDoc(WORKSPACES, id, workspace);
    pending.delete(id);
    reserved.onKept?.(id);
    notify({ workspace: id });
    return workspace;
  }

  async function content(version: Version): Promise<Blob | null> {
    if (version.text !== undefined) return new Blob([version.text], { type: "text/plain;charset=utf-8" });
    if (!version.bytes) return null;
    return version.bytes.store === "image" ? store.getImage(version.bytes.id) : store.getBlob(version.bytes.id);
  }

  /** Keeps `text` with the file's version `n`, as the file is now: the computation may have outlived a rename or a newer version. */
  function keepDerived(file: WorkspaceFile, n: number, text: string): Promise<void> {
    return serial(async () => {
      const value = await store.getDoc<unknown>(filesCollection(file.workspace), file.id);
      if (!isFile(value) || !value.versions.some((v) => v.n === n && v.derived === undefined && v.text === undefined)) return;
      const versions = value.versions.map((v) => (v.n === n ? { ...v, derived: text } : v));
      await store.putDoc(filesCollection(file.workspace), file.id, { ...value, versions });
      notify({ workspace: file.workspace, file: file.id });
    });
  }

  /**
   * Computes a version's text from its bytes, outside the write queue (a long
   * clip takes minutes), and keeps it; what it cannot compute it says. A store
   * that refuses rejects.
   */
  async function compute(file: WorkspaceFile, version: Version, signal: AbortSignal): Promise<Readable> {
    const bytes = await content(version);
    if (!bytes) return { ok: false, error: `${file.name} came from an older chat, which kept no bytes of it to read.` };
    if (!derive) return { ok: false, error: `${file.name}'s text cannot be computed here.` };
    let text: string;
    if (file.kind === "pdf") {
      try {
        text = await derive.pdfText(await bytes.arrayBuffer());
      } catch (err) {
        return { ok: false, error: `${file.name} could not be read as a PDF: ${err instanceof Error ? err.message : String(err)}` };
      }
    } else {
      return { ok: false, error: `${file.name}: audio transcription is not supported by this fork.` };
    }
    if (signal.aborted) throw new Error("stopped");
    await keepDerived(file, version.n, text);
    return { ok: true, text };
  }

  /** Keeps the file with its new version `version`, made by `by`, and says what it wrote. */
  async function commit(file: WorkspaceFile, version: Version, by: Changer, created: boolean): Promise<WriteOutcome> {
    await store.putDoc(filesCollection(file.workspace), file.id, file);
    let workspace: Workspace;
    try {
      workspace = await changed(file.workspace, version.at);
    } catch (err) {
      // The first file of a workspace the store did not keep is not left behind in it.
      if (pending.has(file.workspace)) await store.deleteDoc(filesCollection(file.workspace), file.id).catch(() => {});
      throw err;
    }
    saw(by, file.id, version.n);
    notify({ workspace: file.workspace, file: file.id });
    return { ok: true, written: { workspace: file.workspace, file: file.id, name: file.name, version: version.n }, version, created, workspace };
  }

  /** The file `id` of `workspace` as it is now, with the version `by` adds to it, or why `by` may not change it. */
  async function next(workspace: WorkspaceId, id: string, by: Changer): Promise<{ file: WorkspaceFile; n: number } | { error: string }> {
    const value = await store.getDoc<unknown>(filesCollection(workspace), id);
    if (!isFile(value)) return { error: "The workspace has no such file." };
    const newest = latest(value);
    if (newest.deleted) {
      return { error: isReader(by) ? `${value.name} was deleted by ${authorLabel(newest.author, by.author)} (version ${newest.n}).` : `${value.name} is already deleted.` };
    }
    const refused = guard(by, value);
    return refused ? { error: refused } : { file: value, n: newest.n + 1 };
  }

  return {
    async list() {
      const values = await store.listDocs<unknown>(WORKSPACES);
      return values.filter(isWorkspace).sort((a, b) => b.updated - a.updated);
    },
    get,
    files,
    async file(workspace, id) {
      const value = await store.getDoc<unknown>(filesCollection(workspace), id);
      return isFile(value) ? value : null;
    },
    content,
    create,
    reserve: (name) => reserve(name),
    rename(id, raw) {
      return serial(async () => {
        const name = raw.trim();
        if (!name) return { ok: false as const, error: "A workspace needs a name." };
        const workspace = await get(id);
        if (!workspace) return { ok: false as const, error: "That workspace was deleted." };
        const renamed = { ...workspace, name, updated: now() };
        await store.putDoc(WORKSPACES, id, renamed);
        notify({ workspace: id });
        return { ok: true as const, workspace: renamed };
      });
    },
    remove(id, elsewhere) {
      return serial(async () => {
        const gone = await files(id);
        const others: WorkspaceFile[] = [];
        for (const w of await store.listDocs<unknown>(WORKSPACES)) if (isWorkspace(w) && w.id !== id) others.push(...(await files(w.id)));
        const freed = freedBy(gone, others, elsewhere);
        for (const file of gone) await store.deleteDoc(filesCollection(id), file.id);
        await store.deleteDoc(WORKSPACES, id);
        // A chat it was reserved for gets a new one with its next file.
        for (const [key, reserved] of reservations) if (reserved === id) reservations.delete(key);
        // What the store fails to free stays in it, unnamed; the workspace is gone either way.
        for (const blob of freed.blobs) await store.deleteBlob(blob).catch(() => {});
        for (const image of freed.images) await store.deleteImage(image).catch(() => {});
        for (const file of gone) notify({ workspace: id, file: file.id });
        notify({ workspace: id });
        return { files: gone.length };
      });
    },
    async pictures() {
      const held = new Set<ImageId>();
      for (const w of await store.listDocs<unknown>(WORKSPACES)) if (isWorkspace(w)) for (const id of bytesOf(await files(w.id)).images) held.add(id);
      return held;
    },
    chat(key, binding) {
      return {
        key,
        reader: newReader({ by: "chat", chat: key }, binding.seen),
        agent: (agent) => newReader({ by: "chat", chat: key, agent: { call: agent.call, name: agent.name } }),
        bound: binding.bound,
        workspace() {
          // In the queue of writes: two calls at once reserve one workspace, and a deletion before them is seen.
          return serial(async () => {
            const bound = binding.bound();
            if (bound !== undefined && (await home(bound))) return bound;
            // A reservation whose first file the store refused waits for the next one; once kept, it is the chat's
            // binding, and a chat that moved out of it (#404) gets a new one.
            const reserved = reservations.get(key);
            if (reserved !== undefined && pending.has(reserved)) return reserved;
            // None yet, or the one the chat names is gone: a new one, kept with its first file.
            const workspace = reserve(binding.name(), (id) => binding.onBorn(id, bound));
            reservations.set(key, workspace.id);
            return workspace.id;
          });
        },
      };
    },
    read(reader, file) {
      const version = latest(file);
      saw(reader, file.id, version.n);
      return version;
    },
    writeText(workspace, raw, text, by) {
      return serial(async (): Promise<WriteOutcome> => {
        const name = cleanName(raw);
        if (!name) return { ok: false, error: `"${raw}" is not a file name.` };
        // Nothing is written under a workspace that is gone.
        if (!(await home(workspace))) return { ok: false, error: "That workspace was deleted." };
        const kind = kindOf(name);
        const had = named(await files(workspace), name);
        if (had ? had.kind !== "text" : kind !== "text") {
          return { ok: false, error: `${name} is a ${KIND_LABEL[had?.kind ?? kind]} file, and only a text file can be written whole.` };
        }
        // A deleted file the reader never knew of is a new file to it, and a new file has no guard.
        const refused = had && (present(had) || (isReader(by) && seen.get(by)?.has(had.id))) ? guard(by, had) : null;
        if (refused) return { ok: false, error: refused };
        const version: Version = { n: had ? latest(had).n + 1 : 1, author: authorOf(by), at: now(), text };
        const file: WorkspaceFile = had
          ? { ...had, versions: [...had.versions, version] }
          : { format: WORKSPACE_FORMAT, id: newId(), workspace, name, kind: "text", versions: [version] };
        return commit(file, version, by, !had || !present(had));
      });
    },
    editText(workspace, id, old, replacement, by) {
      return serial(async (): Promise<WriteOutcome> => {
        const found = await next(workspace, id, by);
        if ("error" in found) return { ok: false, error: found.error };
        const { file, n } = found;
        if (file.kind !== "text") return { ok: false, error: `${file.name} is a ${KIND_LABEL[file.kind]} file, and only a text file can be edited.` };
        if (old === "") return { ok: false, error: 'The passage to replace ("old") is empty.' };
        if (old === replacement) return { ok: false, error: 'The passage to replace ("old") and its replacement ("new") are the same: the edit would change nothing.' };
        const text = latest(file).text ?? "";
        let at = occurrences(text, old);
        let swap = [old, replacement];
        // read_file shows a file's lines without their "\r": a passage over several lines of a file with CRLF line endings matches as it would have read.
        if (at.length === 0 && text.includes("\r\n") && old.includes("\n")) {
          swap = [old, replacement].map((s) => s.replace(/\r?\n/g, "\r\n"));
          at = occurrences(text, swap[0]);
        }
        if (at.length === 0) {
          return { ok: false, error: `The passage to replace ("old") occurs nowhere in ${file.name}. Copy it exactly as read_file shows it, without the line numbers.` };
        }
        if (at.length > 1) {
          const lines = [...new Set(at.map((i) => lineAt(text, i)))];
          return {
            ok: false,
            error: `The passage to replace ("old") occurs ${at.length} times in ${file.name}, on ${lines.length === 1 ? "line" : "lines"} ${listed(lines)}: give more of the text around the one to change, so that it occurs once.`,
          };
        }
        const version: Version = {
          n,
          author: authorOf(by),
          at: now(),
          text: text.slice(0, at[0]) + swap[1] + text.slice(at[0] + swap[0].length),
        };
        return commit({ ...file, versions: [...file.versions, version] }, version, by, false);
      });
    },
    add(workspace, incoming, by) {
      return serial(async () => {
        // Nothing is written under a workspace that is gone.
        if (!(await home(workspace))) throw new Error("That workspace was deleted.");
        const taken = (await files(workspace)).map((f) => f.name);
        const written: Written[] = [];
        try {
          for (const item of incoming) {
            const at = now();
            const name = uniqueName(cleanName(item.name) ?? "file", taken);
            taken.push(name);
            const { kind, bytes } = await keep(item);
            const version: Version = {
              n: 1,
              author: authorOf(by),
              at,
              ...(item.text !== undefined ? { text: item.text } : {}),
              ...(bytes ? { bytes } : {}),
              ...(item.derived !== undefined ? { derived: item.derived } : {}),
            };
            const file: WorkspaceFile = { format: WORKSPACE_FORMAT, id: newId(), workspace, name, kind, versions: [version] };
            await store.putDoc(filesCollection(workspace), file.id, file);
            written.push({ workspace, file: file.id, name, version: 1 });
          }
          if (written.length > 0) await changed(workspace, now());
        } catch (err) {
          // All of them or none: the files the store took before it refused one go again, and a workspace born with them is not kept.
          for (const w of written) await store.deleteDoc(filesCollection(workspace), w.file).catch(() => {});
          throw err;
        }
        for (const w of written) saw(by, w.file, w.version);
        for (const w of written) notify({ workspace, file: w.file });
        return written;
      });
    },
    deleteFile(workspace, id, by) {
      return serial(async (): Promise<FileOutcome> => {
        const found = await next(workspace, id, by);
        if ("error" in found) return { ok: false, error: found.error };
        const { file, n } = found;
        const version: Version = { n, author: authorOf(by), at: now(), deleted: true };
        const outcome = await commit({ ...file, versions: [...file.versions, version] }, version, by, false);
        return outcome.ok ? { ok: true, written: outcome.written } : outcome;
      });
    },
    renameFile(workspace, id, raw) {
      return serial(async (): Promise<FileOutcome> => {
        const name = cleanName(raw);
        if (!name) return { ok: false, error: `"${raw.trim()}" is not a file name.` };
        const all = await files(workspace);
        const file = all.find((f) => f.id === id);
        if (!file) return { ok: false, error: "The workspace has no such file." };
        if (!present(file)) return { ok: false, error: `${file.name} is deleted.` };
        const written: Written = { workspace, file: id, name, version: latest(file).n };
        if (name === file.name) return { ok: true, written };
        const holder = all.find((f) => f.name === name);
        if (holder) {
          return { ok: false, error: present(holder) ? `A file named ${name} is already in the workspace.` : `A deleted file named ${name} keeps that name in its history.` };
        }
        await store.putDoc(filesCollection(workspace), id, { ...file, name });
        await touch(workspace, now());
        notify({ workspace, file: id });
        return { ok: true, written };
      });
    },
    restore(workspace, id, n) {
      return serial(async (): Promise<FileOutcome> => {
        const value = await store.getDoc<unknown>(filesCollection(workspace), id);
        if (!isFile(value)) return { ok: false, error: "The workspace has no such file." };
        const old = value.versions.find((v) => v.n === n);
        const newest = latest(value);
        if (!old) return { ok: false, error: `${value.name} has no version ${n}.` };
        if (old.deleted) return { ok: false, error: `Version ${n} of ${value.name} is its deletion, which has no content to restore.` };
        if (old === newest) return { ok: false, error: `Version ${n} is already the newest of ${value.name}.` };
        const version: Version = {
          n: newest.n + 1,
          author: USER,
          at: now(),
          ...(old.text !== undefined ? { text: old.text } : {}),
          ...(old.bytes ? { bytes: old.bytes } : {}),
          ...(old.derived !== undefined ? { derived: old.derived } : {}),
        };
        const outcome = await commit({ ...value, versions: [...value.versions, version] }, version, USER, false);
        return outcome.ok ? { ok: true, written: outcome.written } : outcome;
      });
    },
    readable(file, version, signal) {
      if (version.text !== undefined) return Promise.resolve({ ok: true, text: version.text });
      if (version.derived !== undefined) return Promise.resolve({ ok: true, text: version.derived });
      if (version.deleted || (file.kind !== "pdf" && file.kind !== "audio")) return Promise.resolve({ ok: true, text: undefined });
      const key = `${file.workspace}/${file.id}:${version.n}`;
      let job = deriving.get(key);
      if (!job) {
        const stop = new AbortController();
        const made = { result: compute(file, version, stop.signal), stop, waiting: 0 };
        const done = () => deriving.get(key) === made && deriving.delete(key);
        made.result.then(done, done);
        deriving.set(key, made);
        job = made;
      }
      const shared = job;
      shared.waiting++;
      const stopped: Readable = { ok: false, error: `Stopped before ${file.kind === "audio" ? "the transcript" : "the text"} of ${file.name} was ready.` };
      return new Promise<Readable>((resolve, reject) => {
        const leave = () => {
          // The last read waiting stops the computation: nothing is kept, and the next read starts again.
          if (--shared.waiting === 0) {
            shared.stop.abort();
            if (deriving.get(key) === shared) deriving.delete(key);
          }
          resolve(stopped);
        };
        if (signal?.aborted) return leave();
        signal?.addEventListener("abort", leave, { once: true });
        shared.result.then(
          (out) => {
            signal?.removeEventListener("abort", leave);
            resolve(out);
          },
          (err) => {
            signal?.removeEventListener("abort", leave);
            if (signal?.aborted || shared.stop.signal.aborted) resolve(stopped);
            else reject(err);
          },
        );
      });
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

let page: Workspaces | null = null;

/** The page's one workspace module, on the page's store: one queue of writes and one set of listeners for every chat. */
export function pageWorkspaces(): Workspaces {
  page ??= createWorkspaces(pageStore(), { derive: { pdfText } });
  return page;
}
