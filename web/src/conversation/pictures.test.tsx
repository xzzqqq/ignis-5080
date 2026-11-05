import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { contextUsage } from "../metrics/context.ts";
import { ImageStrip } from "../tools/image/ImageStrip.tsx";
import { Composer } from "./Composer.tsx";
import { UserTurn } from "./UserTurn.tsx";

// The chat's pictures and what can be done with them: a picture the user sent
// or one the picture tool made is marked into the prompt box or opened in the
// Image view; a picture of the box is marked where it is.

const none = () => {};
const actions = { canRerun: true, onSave: none, onFork: none };
const image = { name: "room.jpg", url: "data:image/jpeg;base64,AA==", width: 4, height: 3 };

describe("pictures of the conversation", () => {
  it("offer marking and the Image view on the user's own, when the page takes them", () => {
    const message = { id: 1, role: "user" as const, content: "what is this", reasoning: "", streaming: false, images: [image] };
    const html = renderToStaticMarkup(<UserTurn message={message} actions={actions} onResend={none} onPicture={none} />);
    expect(html).toContain('aria-label="Mark room.jpg"');
    expect(html).not.toContain('aria-label="Open room.jpg in the Image view"');
    expect(renderToStaticMarkup(<UserTurn message={message} actions={actions} onResend={none} />)).not.toContain("Mark room.jpg");
  });

  it("offer the same on a picture the tool made, over the picture as on the user's, and say when it kept to a marked area", () => {
    const run = {
      callId: "c",
      prompt: "a fox",
      references: ["user-1"],
      transparent: false,
      count: 1,
      status: "done" as const,
      made: [{ name: "made-1", url: "data:image/png;base64,AAA", size: "2048x2048", kept: true as const }],
    };
    const html = renderToStaticMarkup(<ImageStrip runs={[run]} onPicture={none} />);
    expect(html).toContain('aria-label="Mark made-1.png"');
    expect(html).not.toContain('aria-label="Open made-1.png in the Image view"');
    expect(html).toContain("Marked area only");
    expect(renderToStaticMarkup(<ImageStrip runs={[run]} />)).not.toContain("Mark made-1.png");
  });

  it("open a picture the tool made full screen, and its whole prompt on asking", () => {
    const run = {
      callId: "c",
      prompt: "a fox in the snow, at dusk",
      references: [],
      transparent: false,
      count: 1,
      status: "done" as const,
      made: [{ name: "made-1", url: "data:image/png;base64,AAA", size: "2048x2048" }],
    };
    const html = renderToStaticMarkup(<ImageStrip runs={[run]} />);
    expect(html).toContain('aria-label="Open made-1 full screen"');
    expect(html).toMatch(/<button[^>]*aria-expanded="false"[^>]*>.*a fox in the snow, at dusk/);
  });

  it("mark a picture of the prompt box where it is, the marked one shown so", () => {
    const props = {
      value: "",
      onChange: none,
      ready: true,
      busy: false,
      streamingHere: false,
      usage: contextUsage([], null, null),
      onSend: none,
      onStop: none,
      canAttach: false,
      attachments: [],
      attachError: null,
      onAttach: none,
      onDetach: none,
      imageError: null,
      onAddImages: none,
      onRemoveImage: none,
      onMarkImage: none,
    };
    const marked = { ...image, markup: { source: new Blob([]), marks: [], keepOutside: true, region: new Blob([]) } };
    const html = renderToStaticMarkup(<Composer {...props} images={[image, marked]} />);
    expect(html).toContain('aria-label="Mark room.jpg"');
    expect(html).toContain('aria-label="Change the marks on room.jpg"');
  });
});
