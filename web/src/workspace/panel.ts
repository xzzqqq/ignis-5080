import { formatBytes } from "../ui/bytes.ts";
import { readUpload } from "./attach.ts";
import { type Author, cleanName, KIND_LABEL, kindOf, latest, present, USER, type Version, type WorkspaceFile } from "./files.ts";
import type { ChatBinding, FileOutcome, Workspaces, WriteOutcome } from "./workspaces.ts";

// What the workspace panel does, apart from how it looks (WorkspacePanel.tsx):
// the tree it shows of the files' names, the size it gives each one, who it
// says made a version, the editor's state, and the user's actions that make a
// file or save one, through the workspace module. A chat with
// no workspace yet gets one born with the first file the user makes or
// brings here, named after the chat, as with an attached file.

/** A folder of the tree, named by the part of the names before a `/`, or a file under the last part of its name. */
export type TreeNode = { kind: "folder"; name: string; path: string; children: TreeNode[] } | { kind: "file"; name: string; file: WorkspaceFile };

const byName = (a: TreeNode, b: TreeNode) => {
  if (a.kind !== b.kind) return a.kind === "folder" ? -1 : 1;
  const x = a.name.toLowerCase();
  const y = b.name.toLowerCase();
  return x < y ? -1 : x > y ? 1 : 0;
};

/** The files in the workspace as a tree of their names split on `/`: folders first, then files, each by name; the deleted files too when `deleted`. */
export function fileTree(files: WorkspaceFile[], deleted = false): TreeNode[] {
  const root: TreeNode[] = [];
  for (const file of deleted ? files : files.filter(present)) {
    const parts = file.name.split("/");
    let level = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const path = parts.slice(0, i + 1).join("/");
      let folder = level.find((n): n is Extract<TreeNode, { kind: "folder" }> => n.kind === "folder" && n.path === path);
      if (!folder) {
        folder = { kind: "folder", name: parts[i], path, children: [] };
        level.push(folder);
      }
      level = folder.children;
    }
    level.push({ kind: "file", name: parts[parts.length - 1], file });
  }
  const sort = (nodes: TreeNode[]): TreeNode[] => {
    nodes.sort(byName);
    for (const n of nodes) if (n.kind === "folder") sort(n.children);
    return nodes;
  };
  return sort(root);
}

/** A version's size as the panel shows it: its bytes, text counted as UTF-8 as a download saves it. */
export function sizeLabel(version: Version): string {
  if (version.text !== undefined) return formatBytes(new TextEncoder().encode(version.text).length);
  if (version.bytes) return formatBytes(version.bytes.size);
  return `${(version.derived ?? "").length.toLocaleString("en-US")} characters of text`;
}

/** The chats the panel names a version's author by, by their keys: whether it is the chat the panel is open in, and its title while the page has it. */
export type ChatNames = { here: (key: string) => boolean; title: (key: string) => string | undefined };

/** Who made a version, as the user reads it: you, this chat or one of its sub-agents, or another chat, by its title while the page has it. */
export function authorName(author: Author, chats: ChatNames): string {
  if (author.by === "user") return "you";
  if (chats.here(author.chat)) return author.agent ? `this chat's sub-agent ${author.agent.name}` : "this chat";
  const title = chats.title(author.chat);
  const chat = title ? `the chat “${title}”` : "another chat";
  return author.agent ? `the sub-agent ${author.agent.name} of ${chat}` : chat;
}

/** What the editor holds: the version it opened or last saved, and the text in its box. */
export type Draft = { base: number; text: string };

/** The editor over the file as it is now: the text it shows, whether that is a change to save, and the newer version made since it opened, if one was. */
export type Editing = Draft & { dirty: boolean; since: Version | null };

/** Two texts the same but for their line endings, which a browser's text box turns into LF. */
const sameText = (a: string, b: string) => a.replace(/\r\n/g, "\n") === b.replace(/\r\n/g, "\n");

/**
 * The editor's state over the file as it is now. While nothing is typed it
 * follows the newest version; once something is, it keeps the text and names
 * the newer version made since, which a Save keeps in the history. A version
 * it saved and the file does not hold yet is not a change.
 */
export function editing(file: WorkspaceFile, draft: Draft): Editing {
  const newest = latest(file);
  const base = file.versions.find((v) => v.n === draft.base);
  if (!base) return { ...draft, dirty: false, since: null };
  if (newest.text !== undefined && sameText(newest.text, draft.text)) return { base: newest.n, text: draft.text, dirty: false, since: null };
  const dirty = !sameText(draft.text, base.text ?? "");
  if (!dirty && newest.text !== undefined) return { base: newest.n, text: newest.text, dirty: false, since: null };
  return { ...draft, dirty, since: newest.n !== draft.base ? newest : null };
}

/** The name a download saves a file under: the last part of its name. */
export const downloadName = (name: string) => name.slice(name.lastIndexOf("/") + 1);

/** An empty text file of the user's, named `raw`; a name another file holds takes a number, as an attached file's does. */
export async function createFile(workspaces: Workspaces, chat: ChatBinding, raw: string): Promise<FileOutcome> {
  if (!raw.trim()) return { ok: false, error: "The file needs a name, like notes.md or site/index.html." };
  const name = cleanName(raw);
  if (!name) return { ok: false, error: `"${raw.trim()}" is not a file name.` };
  const kind = kindOf(name);
  if (kind !== "text") return { ok: false, error: `A new file is a text file, and ${name} names a ${KIND_LABEL[kind]} file.` };
  const [written] = await workspaces.add(await chat.workspace(), [{ name, kind: "text", text: "" }], USER);
  return { ok: true, written };
}

/**
 * The editor's Save: the text the user typed, as a new version of the file
 * authored by the user, through the user's entrance, which the guard never
 * refuses; the model's next change to the file is refused until it reads it
 * again. A browser's text box gives its text with LF line endings, so a file
 * with CRLF ones keeps them.
 */
export function saveText(workspaces: Workspaces, file: WorkspaceFile, text: string): Promise<WriteOutcome> {
  const crlf = latest(file).text?.includes("\r\n") ?? false;
  return workspaces.writeText(file.workspace, file.name, crlf ? text.replace(/\r?\n/g, "\r\n") : text, USER);
}

/** Files the user uploads or drops on the panel, of any kind, as the user's; the names they took in the workspace. */
export async function uploadFiles(workspaces: Workspaces, chat: ChatBinding, files: File[], readPdf?: (data: ArrayBuffer) => Promise<string>): Promise<string[]> {
  if (files.length === 0) return [];
  const incoming = await Promise.all(files.map((f) => readUpload(f, readPdf)));
  const written = await workspaces.add(await chat.workspace(), incoming, USER);
  return written.map((w) => w.name);
}
