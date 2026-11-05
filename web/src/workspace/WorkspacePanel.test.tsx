import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { USER, type Version, type Workspace, type WorkspaceFile } from "./files.ts";
import { authorName, fileTree } from "./panel.ts";
import { idleActions as actions } from "./fixture.ts";
import { FileTree, FileViewer, NameField, PanelView } from "./WorkspacePanel.tsx";

// The workspace panel as it renders: the files as a tree, each file's viewer,
// the actions on each, and what it says when the browser refused a change.
// Clicks are not exercised here: what each action does is panel.test.ts's and
// manage.test.ts's.

const workspace: Workspace = { format: 1, id: "w1", name: "Trip plan", created: 0, updated: 0 };

const at = new Date(2026, 9, 28, 14, 5).getTime();
const text = (t: string, n = 1): Version => ({ n, author: USER, at, text: t });

const file = (name: string, kind: WorkspaceFile["kind"], versions: Version[]): WorkspaceFile => ({ format: 1, id: `f-${name}`, workspace: "w1", name, kind, versions });

const files: WorkspaceFile[] = [
  file("site/index.html", "text", [text("<p>Hi</p>"), { n: 2, author: { by: "chat", chat: "k" }, at, text: "<h1>Rome</h1>" }]),
  file("site/css/main.css", "text", [text("h1{color:red}")]),
  file("notes.md", "text", [text("# Trip")]),
  file("data.zip", "other", [{ n: 1, author: USER, at, bytes: { store: "blob", id: "b1", type: "application/zip", size: 2048 } }]),
  file("gone.txt", "text", [text("x"), { n: 2, author: USER, at, deleted: true }]),
];

/** The chat the panel is open in is "k", the chat that wrote site/index.html's version 2. */
const chats = { here: (key: string) => key === "k", title: () => undefined };
const whom = (author: Version["author"]) => authorName(author, chats);

const byName = (name: string) => files.find((f) => f.name === name)!;

describe("the workspace panel", () => {
  it("lists the workspace's files as a tree of their names, with the panel's actions", () => {
    const html = renderToStaticMarkup(<PanelView workspace={workspace} files={files} chats={chats} onClose={() => {}} actions={actions} />);
    expect(html).toContain('aria-label="Workspace"');
    expect(html).toContain("Trip plan");
    // The folders, then the files of each level, by the last part of their names.
    const order = ["site", "css", "main.css", "index.html", "data.zip", "notes.md"].map((s) => html.indexOf(`>${s}<`));
    expect(order.every((i) => i > -1)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(html).not.toContain("gone.txt");
    expect(html).toMatch(/<button[^>]*>(<svg.*?<\/svg>)?New file<\/button>/);
    expect(html).toMatch(/<input[^>]*type="file"[^>]*multiple=""/);
    expect(html).toContain("Upload");
    expect(html).toContain("or drop files here");
    expect(html).toContain("2 KB");
  });

  it("offers to show the deleted files, with their count", () => {
    const html = renderToStaticMarkup(<PanelView workspace={workspace} files={files} chats={chats} onClose={() => {}} actions={actions} />);
    expect(html).toMatch(/<button[^>]*aria-pressed="false"[^>]*>Show deleted files \(1\)<\/button>/);
    const none = renderToStaticMarkup(<PanelView workspace={workspace} files={files.filter((f) => f.name !== "gone.txt")} chats={chats} onClose={() => {}} actions={actions} />);
    expect(none).not.toContain("Show deleted files");
  });

  it("offers Import zip always and Export zip once there are files, and says what an import made", () => {
    const withFiles = renderToStaticMarkup(<PanelView workspace={workspace} files={files} chats={chats} onClose={() => {}} actions={actions} made="Imported 2 files as the new workspace Trip." />);
    expect(withFiles).toMatch(/Import zip<input[^>]*type="file"[^>]*accept="\.zip,application\/zip"/);
    expect(withFiles).toMatch(/<button[^>]*>Export zip<\/button>/);
    expect(withFiles).toContain("Imported 2 files as the new workspace Trip.");
    const none = renderToStaticMarkup(<PanelView workspace={null} files={[]} chats={chats} onClose={() => {}} actions={actions} />);
    expect(none).toContain("Import zip");
    expect(none).not.toContain("Export zip");
  });

  it("says when the chat has no workspace yet, and still offers to make the first file", () => {
    const html = renderToStaticMarkup(<PanelView workspace={null} files={[]} chats={chats} onClose={() => {}} actions={actions} />);
    expect(html).toContain("No workspace yet");
    expect(html).toContain("The first file you add here, attach or the model writes creates it.");
    expect(html).toMatch(/<button[^>]*>(<svg.*?<\/svg>)?New file<\/button>/);
  });

  it("says when the browser refused a change", () => {
    const html = renderToStaticMarkup(
      <PanelView workspace={workspace} files={files} chats={chats} notice="Not saved in this browser: QuotaExceededError" onClose={() => {}} actions={actions} />,
    );
    expect(html).toMatch(/role="status"[^>]*>Not saved in this browser: QuotaExceededError</);
  });
});

describe("the tree's rows", () => {
  it("offer rename, download and delete for each file", () => {
    const html = renderToStaticMarkup(<FileTree nodes={fileTree(files)} renaming={null} onOpen={() => {}} onRenaming={() => {}} actions={actions} />);
    for (const name of ["site/index.html", "notes.md", "data.zip"]) {
      expect(html).toContain(`aria-label="Rename ${name}"`);
      expect(html).toContain(`aria-label="Download ${name}"`);
      expect(html).toContain(`aria-label="Delete ${name}"`);
    }
  });

  it("show a deleted file as deleted, to open its history, with nothing else to do to it", () => {
    const html = renderToStaticMarkup(<FileTree nodes={fileTree(files, true)} renaming={null} onOpen={() => {}} onRenaming={() => {}} actions={actions} />);
    expect(html).toMatch(/<button[^>]*title="gone.txt, deleted: open its history"[^>]*>gone.txt<\/button><span[^>]*>deleted<\/span>/);
    expect(html).not.toContain('aria-label="Rename gone.txt"');
    expect(html).not.toContain('aria-label="Download gone.txt"');
    expect(html).not.toContain('aria-label="Delete gone.txt"');
    expect(html).toContain('aria-label="Delete notes.md"');
  });

  it("turn the file being renamed into a field holding its whole name", () => {
    const html = renderToStaticMarkup(<FileTree nodes={fileTree(files)} renaming="f-notes.md" onOpen={() => {}} onRenaming={() => {}} actions={actions} />);
    expect(html).toMatch(/<input[^>]*aria-label="New name for notes.md"[^>]*value="notes.md"/);
    expect(html).not.toContain('aria-label="Rename notes.md"');
  });

  it("name a field's refusal beside it", () => {
    const html = renderToStaticMarkup(<NameField label="New name for a.md" initial="a.md" error="A file named b.md is already in the workspace." onSave={async () => null} onCancel={() => {}} />);
    expect(html).toContain("A file named b.md is already in the workspace.");
  });
});

describe("a file's viewer", () => {
  const view = (f: WorkspaceFile) => renderToStaticMarkup(<FileViewer file={f} whom={whom} onBack={() => {}} actions={actions} />);

  it("opens an HTML file's latest version in the sandboxed preview, with full screen and Save as PDF", () => {
    const html = view(byName("site/index.html"));
    expect(html).toMatch(/<iframe[^>]*sandbox="allow-scripts allow-forms"/);
    expect(html).toContain("&lt;h1&gt;Rome&lt;/h1&gt;");
    expect(html).not.toContain("&lt;p&gt;Hi&lt;/p&gt;");
    expect(html).toContain("Save as PDF");
    expect(html).toContain("Expand preview");
    expect(html).toContain("version 2");
    expect(html).toContain("by this chat");
    // The preview first, its editor and its history a tab away.
    expect(html).toMatch(/<button[^>]*role="tab"[^>]*aria-selected="true"[^>]*>Preview<\/button>/);
    expect(html).toMatch(/<button[^>]*role="tab"[^>]*aria-selected="false"[^>]*>Edit<\/button>/);
    expect(html).toMatch(/<button[^>]*role="tab"[^>]*aria-selected="false"[^>]*>History \(2\)<\/button>/);
  });

  it("opens any other text file in its editor, its history a tab away", () => {
    const html = view(byName("site/css/main.css"));
    expect(html).toMatch(/<textarea[^>]*aria-label="The text of site\/css\/main.css"[^>]*>h1\{color:red\}<\/textarea>/);
    expect(html).not.toContain("<iframe");
    expect(html).toContain("by you");
    expect(html).toMatch(/<button[^>]*role="tab"[^>]*aria-selected="true"[^>]*>Edit<\/button>/);
    expect(html).toMatch(/<button[^>]*role="tab"[^>]*aria-selected="false"[^>]*>History \(1\)<\/button>/);
    expect(html).not.toContain(">Preview<");
  });

  it("opens a deleted file on its history, to restore a version from it", () => {
    const html = view(byName("gone.txt"));
    expect(html).toContain("Deleted in version 2, 2026-10-28 14:05, by you.");
    expect(html).toContain('aria-label="Restore version 1 of gone.txt"');
    expect(html).not.toContain("<textarea");
    expect(html).not.toContain('aria-label="Download gone.txt"');
    expect(html).not.toContain('role="tab"');
  });

  it("shows a file the model cannot read by its name and size", () => {
    const html = view(byName("data.zip"));
    expect(html).toContain("data.zip");
    expect(html).toContain("2 KB");
    expect(html).toContain("The model sees its name and size, and cannot read it.");
    expect(html).not.toContain("<pre");
    expect(html).toContain('aria-label="Download data.zip"');
  });

  it("shows a PDF's text, as the model reads it", () => {
    const pdf = file("brief.pdf", "pdf", [{ n: 1, author: USER, at, bytes: { store: "blob", id: "b2", type: "application/pdf", size: 900 }, derived: "[Page 1]\nThe brief" }]);
    const html = view(pdf);
    expect(html).toContain("900 B");
    expect(html).toContain("Its text, as the model reads it");
    expect(html).toContain("[Page 1]\nThe brief");
  });


});
