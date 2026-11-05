import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { contextUsage } from "../metrics/context.ts";
import type { AttachedChip } from "../workspace/attach.ts";
import { Composer } from "./Composer.tsx";

// The chat's prompt box with dictation (spec speech/04): the mic beside Send,
// the strip above the box, Send waiting while a dictation runs, and the hint
// line's sentences.

const none = () => {};

function render(
  _state: Record<string, unknown> = {},
  extra: { value?: string; busy?: boolean; canAttach?: boolean; attachments?: AttachedChip[] } = {},
) {
  return renderToStaticMarkup(
    <Composer
      value={extra.value ?? "Prima di toccare il piano:"}
      onChange={none}
      ready
      busy={extra.busy ?? false}
      streamingHere={false}
      usage={contextUsage([], null, null)}
      onSend={none}
      onStop={none}
      canAttach={extra.canAttach ?? false}
      attachments={extra.attachments ?? []}
      attachError={null}
      onAttach={none}
      onDetach={none}
      images={[]}
      imageError={null}
      onAddImages={none}
      onRemoveImage={none}
    />,
  );
}

/** The Send button's opening tag. */
const send = (html: string) => /<button[^>]*>Send<\/button>/.exec(html)?.[0] ?? "";

describe("the chat's prompt box with dictation", () => {










  it("keeps the other session's streaming note over the idle words", () => {
    expect(render({}, { busy: true })).toContain("A reply is streaming in another session. Send works again once it ends.");
  });
});

describe("the chat's prompt box and the workspace", () => {
  const brief: AttachedChip = { name: "brief.pdf", kind: "pdf", characters: 1200, size: 48_000 };

  it("attaches files into the workspace while the Workspace switch is on, and a file alone is a prompt", () => {
    const html = render({}, { value: "", canAttach: true, attachments: [brief] });
    expect(html).toContain('name="attachments"');
    expect(html).toContain("brief.pdf");
    expect(html).toContain('aria-label="Remove brief.pdf"');
    expect(send(html)).not.toContain('disabled=""');
    expect(send(render({}, { value: "", canAttach: true }))).toContain('disabled=""');
  });

  it("shows an attached PDF by its text and an attached recording by its size", () => {
    const memo: AttachedChip = { name: "standup.m4a", kind: "audio", characters: 0, size: 2_457_600 };
    const html = render({}, { value: "", canAttach: true, attachments: [brief, memo] });
    expect(html).toMatch(/brief\.pdf<\/span><span[^>]*>PDF, 1,200 chars<\/span>/);
    expect(html).toMatch(/standup\.m4a<\/span><span[^>]*>audio, 2\.3 MB<\/span>/);
    expect(html).toContain('aria-label="Remove standup.m4a"');
  });

  it("offers pictures only while the Workspace switch is off", () => {
    const html = render({}, { value: "", canAttach: false, attachments: [brief] });
    expect(html).not.toContain('name="attachments"');
    expect(html).not.toContain("brief.pdf");
    expect(html).toContain('name="images"');
    expect(send(html)).toContain('disabled=""');
  });
});
