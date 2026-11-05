import { type DragEvent, type ReactNode, useState } from "react";
import { usePictureUrl } from "../store/pictureViews.tsx";
import type { ImageId } from "../store/imageStore.ts";
import { caption, fieldLook } from "../ui/classes.ts";
import { isHtmlName } from "../ui/HtmlPreview.tsx";
import { IconClose, IconDownload, IconPencil, IconPlus, IconTrash } from "../ui/icons.tsx";
import { Lightbox } from "../ui/Lightbox.tsx";
import { type Author, KIND_LABEL, latest, present, refusal, timeLabel, USER, type Version, type Workspace, type WorkspaceFile } from "./files.ts";
import { FileEditor, FileHistory, type SaveOutcome } from "./FileHistory.tsx";
import { BUTTON, ExpandButton, Source } from "./looks.tsx";
import { CsvTable, MarkdownPreview, SitePreview, SvgPreview } from "./previews.tsx";
import { sheetAfter, TextFullscreen, textKind } from "./TextSheet.tsx";
import { isCsvName, isMarkdownName, isSvgName } from "./site.ts";
import { authorName, type ChatNames, createFile, downloadName, fileTree, saveText, sizeLabel, type TreeNode, uploadFiles } from "./panel.ts";
import { exportWorkspace, importWorkspace } from "./transfer.ts";
import type { WorkspaceView } from "./useWorkspaces.ts";
import type { ChatBinding, Workspaces } from "./workspaces.ts";

// The Chat view's workspace panel (spec #170): the files of the chat's
// workspace as a tree of their names, each opening in a viewer for its kind,
// and the user's actions on them — a new empty text file, uploads by the
// picker or by a drop, a recording from the microphone (#395), rename, delete
// and download; an audio file plays beside its transcript; a text file's
// edits by hand, and any file's history with its diff and restore, deleted
// files' included (FileHistory.tsx). Everything goes through the workspace
// module as the user's; a chat with no workspace yet gets one with the first
// file made here. What the browser refuses, the panel says.

/** What the panel's controls do; a name a field sends back comes with the refusal, or null once done. */
export type PanelActions = {
  create: (name: string) => Promise<string | null>;
  upload: (files: File[]) => void;
  /** Saves the workspace as a zip. */
  exportZip: () => void;
  /** Makes a new workspace of a zip; the refusal, or what it made. */
  importZip: (zip: File) => void;
  rename: (file: string, name: string) => Promise<string | null>;
  remove: (file: string) => void;
  download: (file: WorkspaceFile) => void;
  /** The editor's Save: the text as a new version of the user's. */
  save: (file: WorkspaceFile, text: string) => Promise<SaveOutcome>;
  /** Restores version `n` of a file as its newest. */
  restore: (file: string, n: number) => void;
  /** A version's bytes, for a player. */
  content: (version: Version) => Promise<Blob | null>;
  /** Computes the text the model reads of a file's latest version, a recording's transcript, and keeps it; why it could not, or null. */
  derive: (file: WorkspaceFile) => Promise<string | null>;
};

/** An icon button's shape; its hover colour is the caller's (two `hover:text-*` on one element, and Tailwind's order picks). */
const ICON_SHAPE = "flex size-7 shrink-0 items-center justify-center text-ash";

const ICON_BUTTON = `${ICON_SHAPE} hover:text-ink`;

/** The panel for the chat's workspace, reading and changing it through the workspace module. */
export function WorkspacePanel({
  workspaces,
  view,
  chat,
  chats,
  onClose,
}: {
  workspaces: Workspaces;
  /** The chat's workspace and its files, as useWorkspace reads them. */
  view: WorkspaceView;
  /** The chat's hold on its workspace: the first file made here gives it one. */
  chat: () => ChatBinding;
  /** The chat the panel is open in, and the page's chats by their keys: who made each version. */
  chats: ChatNames;
  /** Whether the speech model can serve, from the models listing: a recording's microphone and transcript need it. */
  onClose: () => void;
}) {
  const [notice, setNotice] = useState<string | null>(null);
  const [made, setMade] = useState<string | null>(null);
  const id = view.workspace?.id;
  // A recording under way when the panel closes is saved, not lost.

  /** `job`, with what the store refused said in the panel. */
  async function attempt<T>(job: () => Promise<T>): Promise<T | { refused: string }> {
    setNotice(null);
    setMade(null);
    try {
      return await job();
    } catch (err) {
      const refused = refusal(err);
      setNotice(refused);
      return { refused };
    }
  }

  const fieldError = (out: { ok: true } | { ok: false; error: string } | { refused: string }) =>
    "refused" in out ? out.refused : out.ok ? null : out.error;

  const actions: PanelActions = {
    create: async (name) => fieldError(await attempt(() => createFile(workspaces, chat(), name))),
    upload: (files) => void attempt(() => uploadFiles(workspaces, chat(), files)),
    exportZip: () =>
      void attempt(async () => {
        if (!view.workspace) return;
        const out = await exportWorkspace(workspaces, view.workspace, view.files);
        save(out.zip, out.name);
        if (out.left.length > 0) setNotice(`Left out of the zip, because an older chat kept no bytes of them: ${out.left.join(", ")}.`);
      }),
    importZip: (zip) =>
      void attempt(async () => {
        const out = await importWorkspace(workspaces, zip);
        if (!out.ok) setNotice(out.error);
        else setMade(`Imported ${out.files} ${out.files === 1 ? "file" : "files"} as the new workspace ${out.workspace.name}. Pick it in the Workspace menu of a chat that has none yet.`);
      }),
    rename: async (file, name) => (id ? fieldError(await attempt(() => workspaces.renameFile(id, file, name))) : null),
    remove: (file) => {
      if (!id) return;
      void attempt(() => workspaces.deleteFile(id, file, USER)).then((out) => {
        if (!("refused" in out) && !out.ok) setNotice(out.error);
      });
    },
    download: (file) =>
      void attempt(async () => {
        const blob = await workspaces.content(latest(file));
        if (blob) save(blob, downloadName(file.name));
        else setNotice(`${file.name} came from an older chat, which kept only its text: there are no bytes to download.`);
      }),
    save: async (file, text) => {
      const out = await attempt(() => saveText(workspaces, file, text));
      return "refused" in out ? { ok: false, error: out.refused } : out.ok ? { ok: true, version: out.written.version } : out;
    },
    restore: (file, n) => {
      if (!id) return;
      void attempt(() => workspaces.restore(id, file, n)).then((out) => {
        if (!("refused" in out) && !out.ok) setNotice(out.error);
      });
    },
    content: (version) => workspaces.content(version),
    derive: async (file) => {
      const out = await attempt(() => workspaces.readable(file, latest(file)));
      return "refused" in out ? out.refused : out.ok ? null : out.error;
    },
  };

  return (
    <PanelView
      workspace={view.workspace}
      files={view.files}
      chats={chats}
      notice={notice}
      made={made}
      onClose={onClose}
      actions={actions}
    />
  );
}

function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** The panel as it renders: undefined while the workspace is read, null when the chat has none yet. */
export function PanelView({
  workspace,
  files,
  chats,
  notice,
  made,
  onClose,
  actions,
}: {
  workspace: Workspace | null | undefined;
  files: WorkspaceFile[];
  chats: ChatNames;
  notice?: string | null;
  /** What an import made, said beside the notice. */
  made?: string | null;
  onClose: () => void;
  actions: PanelActions;
  /** Record, and the recording under way. */
}) {
  const [opened, setOpened] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [showDeleted, setShowDeleted] = useState(false);
  // A file deleted while open stays open, on its history.
  const open = files.find((f) => f.id === opened);
  const deleted = files.filter((f) => !present(f)).length;
  const tree = fileTree(files, showDeleted);
  const whom = (author: Author) => authorName(author, chats);

  const drag = (e: DragEvent) => {
    if (![...e.dataTransfer.types].includes("Files")) return;
    e.preventDefault();
    setOver(true);
  };

  return (
    <aside
      aria-label="Workspace"
      className={`fixed inset-y-0 right-0 z-40 flex w-[22rem] max-w-full flex-col border-l bg-ground 2xl:static 2xl:z-auto ${over ? "border-ember" : "border-line"}`}
      onDragOver={drag}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const dropped = Array.from(e.dataTransfer.files);
        if (dropped.length > 0) actions.upload(dropped);
      }}
    >
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className={caption}>Workspace</p>
          <p className="truncate font-display text-sm font-semibold text-ink">
            {workspace ? workspace.name : workspace === null ? "No workspace yet" : "Reading…"}
          </p>
        </div>
        <button type="button" aria-label="Close the workspace" title="Close the workspace" className={ICON_BUTTON} onClick={onClose}>
          <IconClose />
        </button>
      </div>

      {open ? (
        <FileViewer key={open.id} file={open} whom={whom} onBack={() => setOpened(null)} actions={actions} files={files} />
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className={BUTTON} onClick={() => setCreating(true)}>
              <IconPlus />
              New file
            </button>
            <label className={`${BUTTON} cursor-pointer`}>
              Upload
              <input
                type="file"
                name="workspace-files"
                multiple
                className="sr-only"
                onChange={(e) => {
                  const picked = Array.from(e.currentTarget.files ?? []);
                  e.currentTarget.value = "";
                  if (picked.length > 0) actions.upload(picked);
                }}
              />
            </label>
            <label className={`${BUTTON} cursor-pointer`} title="Make a new workspace of a zip">
              Import zip
              <input
                type="file"
                name="workspace-zip"
                accept=".zip,application/zip"
                className="sr-only"
                onChange={(e) => {
                  const picked = e.currentTarget.files?.[0];
                  e.currentTarget.value = "";
                  if (picked) actions.importZip(picked);
                }}
              />
            </label>
            {tree.length > 0 && (
              <button type="button" className={BUTTON} title="Save the workspace as a zip" onClick={actions.exportZip}>
                Export zip
              </button>
            )}
            <span className="text-xs text-ash">or drop files here</span>
            {deleted > 0 && (
              <button
                type="button"
                aria-pressed={showDeleted}
                className="ml-auto shrink-0 font-display text-xs text-ash hover:text-ink"
                onClick={() => setShowDeleted((s) => !s)}
              >
                {showDeleted ? "Hide deleted files" : `Show deleted files (${deleted})`}
              </button>
            )}
          </div>
          {creating && (
            <NameField
              label="Name of the new file"
              initial="notes.md"
              onSave={async (name) => {
                const error = await actions.create(name);
                if (!error) setCreating(false);
                return error;
              }}
              onCancel={() => setCreating(false)}
            />
          )}
          {workspace === null && <p className="text-xs leading-snug text-ash">The first file you add here, attach or the model writes creates it.</p>}
          {workspace && tree.length === 0 && <p className="text-xs text-ash">No files.</p>}
          <FileTree nodes={tree} renaming={renaming} onOpen={setOpened} onRenaming={setRenaming} actions={actions} />
        </div>
      )}

      {made && (
        <p role="status" className="border-t border-line px-4 py-2 text-xs leading-snug text-ash">
          {made}
        </p>
      )}
      {notice && (
        <p role="status" className="border-t border-line px-4 py-2 text-xs leading-snug text-fault">
          {notice}
        </p>
      )}
    </aside>
  );
}

/** The files as a tree: a folder per part of the names before a `/`, open, and a row per file with its actions. */
export function FileTree({
  nodes,
  renaming,
  onOpen,
  onRenaming,
  actions,
}: {
  nodes: TreeNode[];
  /** The file whose name is being edited. */
  renaming: string | null;
  onOpen: (file: string) => void;
  onRenaming: (file: string | null) => void;
  actions: PanelActions;
}) {
  if (nodes.length === 0) return null;
  return (
    <ul className="flex flex-col gap-0.5">
      {nodes.map((node) =>
        node.kind === "folder" ? (
          <li key={`d:${node.path}`}>
            <details open>
              <summary className="cursor-pointer py-1 font-display text-xs font-semibold text-ash hover:text-ink">{node.name}</summary>
              <div className="ml-2 border-l border-line pl-2">
                <FileTree nodes={node.children} renaming={renaming} onOpen={onOpen} onRenaming={onRenaming} actions={actions} />
              </div>
            </details>
          </li>
        ) : (
          <li key={node.file.id}>
            {renaming === node.file.id ? (
              <NameField
                label={`New name for ${node.file.name}`}
                initial={node.file.name}
                onSave={async (name) => {
                  const error = await actions.rename(node.file.id, name);
                  if (!error) onRenaming(null);
                  return error;
                }}
                onCancel={() => onRenaming(null)}
              />
            ) : (
              <FileRow file={node.file} name={node.name} onOpen={() => onOpen(node.file.id)} onRename={() => onRenaming(node.file.id)} actions={actions} />
            )}
          </li>
        ),
      )}
    </ul>
  );
}

function FileRow({ file, name, onOpen, onRename, actions }: { file: WorkspaceFile; name: string; onOpen: () => void; onRename: () => void; actions: PanelActions }) {
  if (!present(file)) {
    return (
      <div className="flex min-w-0 items-center gap-1 hover:bg-surface">
        <button type="button" title={`${file.name}, deleted: open its history`} className="min-w-0 flex-1 truncate py-1 pl-1 text-left text-sm text-ash line-through" onClick={onOpen}>
          {name}
        </button>
        <span className="shrink-0 pr-1 font-display text-xs text-ash">deleted</span>
      </div>
    );
  }
  return (
    <div className="flex min-w-0 items-center gap-1 hover:bg-surface">
      <button type="button" title={file.name} className="min-w-0 flex-1 truncate py-1 pl-1 text-left text-sm text-ink" onClick={onOpen}>
        {name}
      </button>
      <span className="shrink-0 font-display text-xs tabular-nums text-ash">{sizeLabel(latest(file))}</span>
      <button type="button" aria-label={`Rename ${file.name}`} title="Rename" className={ICON_BUTTON} onClick={onRename}>
        <IconPencil />
      </button>
      <button type="button" aria-label={`Download ${file.name}`} title="Download" className={ICON_BUTTON} onClick={() => actions.download(file)}>
        <IconDownload />
      </button>
      <button type="button" aria-label={`Delete ${file.name}`} title="Delete" className={`${ICON_SHAPE} hover:text-fault`} onClick={() => actions.remove(file.id)}>
        <IconTrash />
      </button>
    </div>
  );
}

/** A name being typed: Enter or Save sends it, Escape or Cancel drops it; a refusal stays beside it. */
export function NameField({
  label,
  initial,
  error: refused = null,
  onSave,
  onCancel,
}: {
  label: string;
  initial: string;
  error?: string | null;
  onSave: (name: string) => Promise<string | null>;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initial);
  const [error, setError] = useState<string | null>(refused);
  const submit = async () => setError(await onSave(value));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <input
          aria-label={label}
          value={value}
          autoFocus
          spellCheck={false}
          className={`min-w-0 flex-1 ${fieldLook} py-1`}
          onChange={(e) => setValue(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
            if (e.key === "Escape") onCancel();
          }}
        />
        <button type="button" className="shrink-0 font-display text-xs font-semibold text-ember hover:text-ink" onClick={() => void submit()}>
          Save
        </button>
        <button type="button" className="shrink-0 font-display text-xs text-ash hover:text-ink" onClick={onCancel}>
          Cancel
        </button>
      </div>
      {error && <p className="text-xs leading-snug text-fault">{error}</p>}
    </div>
  );
}

type ViewerTab = "preview" | "edit" | "view" | "history";

const TAB_LABEL: Record<Exclude<ViewerTab, "history">, string> = { preview: "Preview", edit: "Edit", view: "File" };

/**
 * A file opened from the tree, its latest version in the viewer for its kind,
 * a text file's in its editor (an HTML file's a tab after its preview), and
 * its history a tab away. A deleted file opens on its history alone.
 */
export function FileViewer({ file, whom, onBack, actions, files = [] }: { file: WorkspaceFile; whom: (author: Author) => string; onBack: () => void; actions: PanelActions; files?: WorkspaceFile[] }) {
  const v = latest(file);
  const editable = v.text !== undefined;
  const drawn = editable && isDrawnName(file.name);
  const tabs: ViewerTab[] = v.deleted ? ["history"] : drawn ? ["preview", "edit", "history"] : editable ? ["edit", "history"] : ["view", "history"];
  const [chosen, setTab] = useState<ViewerTab>(tabs[0]);
  const tab = tabs.includes(chosen) ? chosen : tabs[0];
  // The name the full screen opened on; it closes for good once the file is deleted or renamed (sheetAfter).
  const [expanded, setExpanded] = useState<string | null>(null);
  const shown = sheetAfter(expanded, file);
  if (shown !== expanded) setExpanded(shown);
  const kind = editable ? textKind(file.name) : null;
  // The full screen opens from the drawn preview, or from the source of a text with no preview.
  const expandable = kind !== null && !v.deleted && (drawn ? tab === "preview" : tab === "edit");
  let body: ReactNode = null;
  if (v.text !== undefined) {
    body = textBody(file, v.text, files, actions);
  } else if (file.kind === "pdf" && v.derived !== undefined) {
    body = (
      <>
        <p className={caption}>Its text, as the model reads it</p>
        <Source text={v.derived} />
      </>
    );
  } else if (file.kind === "audio") {
    body = <p className="text-xs text-ash">Audio transcription is not available in this fork.</p>;
  } else if (file.kind === "picture" && v.bytes?.store === "image") {
    body = <PictureView id={v.bytes.id} name={file.name} width={v.bytes.width} height={v.bytes.height} />;
  } else if (file.kind === "other") {
    body = <p className="text-xs leading-snug text-ash">The model sees its name and size, and cannot read it.</p>;
  }
  const history = <FileHistory file={file} whom={whom} onRestore={(n) => actions.restore(file.id, n)} />;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-4 py-3">
      <div className="flex items-center gap-2">
        <button type="button" className="font-display text-xs font-semibold text-ash hover:text-ink" onClick={onBack}>
          ← All files
        </button>
        {!v.deleted && (
          <button type="button" aria-label={`Download ${file.name}`} title="Download" className={`${ICON_BUTTON} ml-auto`} onClick={() => actions.download(file)}>
            <IconDownload />
          </button>
        )}
      </div>
      <p className={`font-display text-sm font-semibold break-all ${v.deleted ? "text-ash line-through" : "text-ink"}`}>{file.name}</p>
      <p className="text-xs leading-snug text-ash">
        {v.deleted
          ? `Deleted in version ${v.n}, ${timeLabel(v.at)}, by ${whom(v.author)}. Restore a version to bring it back.`
          : `${KIND_LABEL[file.kind]}, ${sizeLabel(v)}; version ${v.n}, changed ${timeLabel(v.at)} by ${whom(v.author)}`}
      </p>
      {tabs.length > 1 && (
        <div role="tablist" aria-label={file.name} className="flex items-center gap-3 border-b border-line">
          {tabs.map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={t === tab}
              className={`-mb-px border-b-2 pb-1 font-display text-xs font-semibold ${t === tab ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
              onClick={() => setTab(t)}
            >
              {t === "history" ? `History (${file.versions.length})` : TAB_LABEL[t]}
            </button>
          ))}
        </div>
      )}
      {expandable && (
        <div className="flex justify-end">
          <ExpandButton onClick={() => setExpanded(file.name)} />
        </div>
      )}
      {shown !== null && kind !== null && v.text !== undefined && <TextFullscreen name={file.name} version={v.n} text={v.text} kind={kind} onClose={() => setExpanded(null)} />}
      {/* The editor stays mounted while another tab shows: what is typed in it waits there. */}
      {editable && (
        <div hidden={tab !== "edit"}>
          <FileEditor file={file} whom={whom} onSave={actions.save} />
        </div>
      )}
      {tab === "history" ? history : tab !== "edit" && body}
    </div>
  );
}

/** A picture of the workspace from the page's store, at the panel's width; a click opens it full screen. */
function PictureView({ id, name, width, height }: { id: ImageId; name: string; width?: number; height?: number }) {
  const url = usePictureUrl()(id, "full");
  const [full, setFull] = useState(false);
  return (
    <>
      {width && height && (
        <p className="text-xs text-ash tabular-nums">
          {width} × {height} pixels
        </p>
      )}
      <button type="button" aria-label={`Open ${name} full screen`} title="Full screen" className="block cursor-zoom-in bg-surface" onClick={() => setFull(true)} disabled={!url}>
        {url ? (
          <img src={url} alt={name} className="stage-checker block max-h-[60vh] max-w-full object-contain" />
        ) : (
          <span role="img" aria-label={name} className="block h-40 w-full bg-line/60" />
        )}
      </button>
      {full && url && (
        <Lightbox onClose={() => setFull(false)}>
          <img src={url} alt={name} className="stage-checker block max-h-[calc(100dvh-2rem)] max-w-[calc(100vw-2rem)] object-contain" />
        </Lightbox>
      )}
    </>
  );
}

/** Whether a text file has a drawn preview: a page, Markdown, SVG or CSV. */
const isDrawnName = (name: string) => isHtmlName(name) || isMarkdownName(name) || isSvgName(name) || isCsvName(name);

/** A text file in the viewer for its name: a page with its site, Markdown, SVG or CSV drawn, anything else as text. */
function textBody(file: WorkspaceFile, text: string, files: WorkspaceFile[], actions: PanelActions): ReactNode {
  if (isHtmlName(file.name)) return <SitePreview file={file} text={text} files={files} content={actions.content} />;
  if (isMarkdownName(file.name)) return <MarkdownPreview text={text} />;
  if (isSvgName(file.name)) return <SvgPreview text={text} name={file.name} />;
  if (isCsvName(file.name)) return <CsvTable text={text} />;
  return null;
}
