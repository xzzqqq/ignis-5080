import { describe, expect, it } from "vitest";
import type { Settings } from "../api/request.ts";
import { createMemoryStore } from "../store/memory.ts";
import { type LocalRun, runLocalCalls } from "../tools/local/local.ts";
import { USER } from "./files.ts";
import type { WorkspaceAccess } from "./tools.ts";
import { createWorkspaces, type Deriving } from "./workspaces.ts";

// The text the model reads of a file's bytes (#395): an audio file's
// transcript, from the server's transcription route, and a PDF's text, each
// computed on the first read of a version and kept with it. The route and
// pdf.js are injected, so no network and no browser are needed.

const settings: Settings = {
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

function deriving() {
  const asked = { pdf: 0 };
  const derive: Deriving = { pdfText: async () => { asked.pdf++; return "[Page 1]\nThe brief"; } };
  return { asked, derive };
}

function setup(derive: Deriving) {
  let n = 0;
  const workspaces = createWorkspaces(createMemoryStore(), { newId: () => `id${++n}`, derive });
  let bound: string | undefined;
  const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "Errands", onBorn: (id) => (bound = id) });
  const access: WorkspaceAccess = { workspaces, chat };
  return { workspaces, access, chat };
}

let callN = 0;

async function read(access: WorkspaceAccess, name: string, signal = new AbortController().signal): Promise<LocalRun> {
  const call = { id: `c${++callN}`, name: "read_file", arguments: JSON.stringify({ name }) };
  const [ran] = await runLocalCalls([call], { settings, signal, jsSafetyCheck: false, workspace: access }, () => {});
  return ran;
}



describe("read_file of a PDF", () => {


  it("reads the text kept when it was attached, and computes nothing", async () => {
    const { derive, asked } = deriving();
    const { workspaces, access } = setup(derive);
    const id = await access.chat.workspace();
    await workspaces.add(id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob([new Uint8Array([1])]), derived: "[Page 1]\nKept" }], USER);
    expect((await read(access, "brief.pdf")).result).toBe("1\t[Page 1]\n2\tKept\n\n[brief.pdf, version 1: lines 1–2 of 2.]");
    expect(asked.pdf).toBe(0);
  });

  it("says what pdf.js could not read", async () => {
    const { derive } = deriving();
    const { workspaces, access } = setup({ ...derive, pdfText: async () => Promise.reject(new Error("Invalid PDF structure.")) });
    const id = await access.chat.workspace();
    await workspaces.add(id, [{ name: "broken.pdf", kind: "pdf", blob: new Blob([new Uint8Array([1])]) }], USER);
    expect((await read(access, "broken.pdf")).error).toBe("broken.pdf could not be read as a PDF: Invalid PDF structure.");
  });
});
