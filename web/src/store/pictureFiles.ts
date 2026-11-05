import type { Seed } from "../api/seed.ts";
import type { ImageProgress } from "../api/images.ts";
type ImageBackground = "auto" | "transparent" | "opaque";

export function progressShare(progress: ImageProgress, n: number): number {
  const images = Math.max(1, n);
  const perImage = progress.phase === "denoising" || progress.phase === "decoding";
  if (!perImage) return 0;
  const within = progress.phase === "decoding" ? 1 : progress.total > 0 ? (progress.step - 1) / progress.total : 0;
  return Math.min(1, (progress.image + Math.min(1, Math.max(0, within))) / images);
}

export type GalleryItem = {
  id: number;
  /** The PNG as a data URL, built once when the picture arrives: the gallery re-renders on every progress event. */
  url: string;
  /** The owner's words: a transparent picture's without the wording `transparentPrompt` added. */
  prompt: string;
  revisedPrompt?: string;
  size: string;
  /** The image's own seed (the job's seed plus its index), when the server said. */
  seed?: Seed;
  edit: boolean;
  /** The background the request chose, when it chose one other than the model's own. */
  background?: Exclude<ImageBackground, "auto">;
  /** Laid over the original through the marked area: outside it, the original's pixels. */
  kept?: true;
};

export function fileFromDataUrl(url: string, name: string): File {
  const match = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (!match) throw new Error("not a data URL");
  const type = match[1] || "application/octet-stream";
  const bytes = match[2] ? Uint8Array.from(atob(match[3]), (c) => c.charCodeAt(0)) : new TextEncoder().encode(decodeURIComponent(match[3]));
  return new File([bytes], name, { type });
}

export const sidesOf = (size: string) => size.replace(/^(\d+)x(\d+)$/, "$1 × $2");

export const downloadName = (item: GalleryItem) => `ignis-${item.seed ?? `image-${item.id}`}-${item.size}.png`;

export function downloadImage(item: GalleryItem, doc: Pick<Document, "createElement"> = document): void {
  const anchor = doc.createElement("a");
  anchor.href = item.url;
  anchor.download = downloadName(item);
  anchor.click();
}
