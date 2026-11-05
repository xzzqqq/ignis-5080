// Legacy pictures in existing chat sessions; this fork cannot generate them.
import type { Written } from "../../workspace/workspaces.ts";
import type { Seed } from "../../api/seed.ts";
import type { ImageProgress } from "../../api/images.ts";

export type ImageStatus = "queued" | "switching" | "making" | "returning" | "done" | "failed" | "stopped";

export type MadePicture = {
  name: string;
  /** The picture itself, when the workspace does not hold it. */
  url?: string;
  /** The workspace file and version it landed as, which the reply shows. */
  file?: Written;
  /** Why the browser's store did not keep it in the workspace. */
  unkept?: string;
  size: string;
  seed?: Seed;
  revisedPrompt?: string;
  /** Laid over the marked picture it edits, through the marked area. */
  kept?: true;
};

export type ImageRun = {
  callId: string;
  prompt: string;
  references: string[];
  aspect?: string;
  transparent: boolean;
  count: number;
  seed?: Seed;
  status: ImageStatus;
  /** What the call is waiting on, in words: the switch, a phase, a step. */
  line?: string;
  progress?: ImageProgress;
  made?: MadePicture[];
  error?: string;
  /** The seed asked made one of the references, so it was left to the server. */
  seedDropped?: true;
};

export function imageSummary(runs: ImageRun[]): string {
  const pictures = runs.reduce((n, r) => n + (r.made?.length ?? 0), 0);
  const failed = runs.filter((r) => r.status === "failed").length;
  const busy = runs.some((r) => r.status !== "done" && r.status !== "failed" && r.status !== "stopped");
  return [
    `${pictures} ${pictures === 1 ? "picture" : "pictures"}`,
    busy ? "under way" : "",
    failed ? `${failed} failed` : "",
  ]
    .filter(Boolean)
    .join(", ");
}
