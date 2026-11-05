import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StaticPictures } from "../store/pictureViews.tsx";
import { ImageStrip } from "../tools/image/ImageStrip.tsx";
import type { ImageRun } from "../tools/image/image.ts";
import { LocalStrip } from "../tools/local/LocalStrip.tsx";
import { USER, type Version, type WorkspaceFile } from "./files.ts";
import { StaticWorkspaceFiles } from "./WorkspaceFiles.tsx";
import { authorName } from "./panel.ts";
import { idleActions as actions } from "./fixture.ts";
import { FileViewer } from "./WorkspacePanel.tsx";

// Pictures of the workspace as the page shows them (#394): the panel's viewer,
// a reply's picture shown from the workspace file its make_image call made,
// and the row of a read_file call that read one.

const at = new Date(2026, 9, 28, 14, 5).getTime();

const picture = (n: number, id: string): Version => ({ n, author: { by: "chat", chat: "k" }, at, bytes: { store: "image", id, type: "image/png", size: 2048, width: 2752, height: 1536 } });

const file = (versions: Version[]): WorkspaceFile => ({ format: 1, id: "f1", workspace: "w1", name: "made-1.png", kind: "picture", versions });

const whom = (author: Version["author"]) => authorName(author, { here: (key) => key === "k", title: () => undefined });

describe("the panel's picture viewer", () => {
  it("shows the picture's latest version from the page's store, with its size in pixels, and opens it full screen", () => {
    const html = renderToStaticMarkup(
      <StaticPictures urls={{ "full:p2": "blob:two" }}>
        <FileViewer file={file([picture(1, "p1"), picture(2, "p2")])} whom={whom} onBack={() => {}} actions={actions} />
      </StaticPictures>,
    );
    expect(html).toMatch(/<img[^>]*src="blob:two"/);
    expect(html).not.toContain("blob:one");
    expect(html).toContain("2752 × 1536");
    expect(html).toContain('aria-label="Open made-1.png full screen"');
    expect(html).toContain("version 2");
  });

  it("holds the picture's place while its bytes are read", () => {
    const html = renderToStaticMarkup(
      <StaticPictures urls={{}}>
        <FileViewer file={file([picture(1, "p1")])} whom={whom} onBack={() => {}} actions={actions} />
      </StaticPictures>,
    );
    expect(html).not.toContain("<img");
    expect(html).toContain('role="img"');
  });
});

describe("a reply's picture made into the workspace", () => {
  const run = (version: number): ImageRun => ({
    callId: "c1",
    prompt: "a fox",
    references: [],
    transparent: false,
    count: 1,
    status: "done",
    made: [{ name: "made-1", size: "2752x1536", seed: "41", file: { workspace: "w1", file: "f1", name: "made-1.png", version } }],
  });

  const render = (r: ImageRun, files: WorkspaceFile[]) =>
    renderToStaticMarkup(
      <StaticWorkspaceFiles files={files}>
        <StaticPictures urls={{ "full:p1": "blob:one", "full:p2": "blob:two" }}>
          <ImageStrip runs={[r]} />
        </StaticPictures>
      </StaticWorkspaceFiles>,
    );

  it("is shown from the workspace, the version its call made, and never from a data URL", () => {
    const html = render(run(1), [file([picture(1, "p1"), { n: 2, author: USER, at, deleted: true }])]);
    expect(html).toMatch(/<img[^>]*src="blob:one"/);
    expect(html).not.toContain("data:");
    expect(html).toContain("made-1.png in the workspace");
    expect(html).toContain("Seed 41");
    expect(html).toContain(">PNG<");
  });

  it("is shown from its own workspace once the chat moved to another", () => {
    const elsewhere: WorkspaceFile = { ...file([picture(1, "p2")]), workspace: "w2" };
    expect(render(run(1), [elsewhere, file([picture(1, "p1")])])).toMatch(/<img[^>]*src="blob:one"/);
  });

  it("says so once its workspace is gone", () => {
    const html = render(run(1), []);
    expect(html).not.toContain("<img");
    expect(html).toContain("made-1.png is no longer in the workspace.");
  });

  it("is shown from its data URL in an older chat, as before", () => {
    const older: ImageRun = { ...run(1), made: [{ name: "made-1", url: "data:image/png;base64,AAA", size: "2752x1536" }] };
    expect(render(older, [])).toMatch(/<img[^>]*src="data:image\/png;base64,AAA"/);
  });
});

describe("a read_file call that read a picture", () => {
  const read = (seen: boolean) =>
    renderToStaticMarkup(<LocalStrip runs={[{ callId: "c1", tool: "read_file", status: "done", picture: { name: "photo.png", seen }, result: "[photo.png]" }]} />);

  it("says whether the model was shown the picture or only told of it", () => {
    expect(read(true)).toContain("photo.png, seen");
    expect(read(false)).toContain("photo.png, as text: the model has no vision");
  });
});
