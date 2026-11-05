import { useState } from "react";
import type { ChatPicture, PictureAction } from "../../conversation/images.ts";
import { PictureTools } from "../../conversation/UserTurn.tsx";
import { usePictureUrl } from "../../store/pictureViews.tsx";
import { downloadImage, progressShare, sidesOf } from "../../store/pictureFiles.ts";
import { IconChevron, IconDownload } from "../../ui/icons.tsx";
import { Lightbox } from "../../ui/Lightbox.tsx";
import { versionOf } from "../../workspace/pictures.ts";
import { useWorkspaceFile } from "../../workspace/WorkspaceFiles.tsx";
import type { Written } from "../../workspace/workspaces.ts";
import { StatusMark } from "../agents/status.tsx";
import { type ImageRun, type ImageStatus, imageSummary, type MadePicture } from "./image.ts";

// The pictures a reply asked for, inside the reply: each call on the kiln's
// dark plate, as on the Image view's stage — the switch and the steps while
// it works, then the pictures with their names, which is how the model and
// the user both refer to them afterwards. A picture that landed in the chat's
// workspace is shown from there, the version the call made; an older chat's
// from the data URL its run kept.

const STATUS_WORDS: Record<ImageStatus, string> = {
  queued: "Waiting for the other tools",
  switching: "Switching to the image model",
  making: "Making",
  returning: "Loading the text model again",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

const markOf = (status: ImageStatus) =>
  status === "switching" || status === "making" || status === "returning" ? "running" : status;

type OnPicture = (picture: ChatPicture, action: PictureAction) => void;

export function ImageStrip({ runs, onPicture }: { runs: ImageRun[]; onPicture?: OnPicture }) {
  return (
    <section aria-label="Pictures" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Pictures</h3>
        <p className="font-display text-xs text-ash">{imageSummary(runs)}</p>
      </div>
      <ul className="flex flex-col gap-3">
        {runs.map((run, i) => (
          <PictureCall key={`${i}:${run.callId}`} run={run} onPicture={onPicture} />
        ))}
      </ul>
    </section>
  );
}

function PictureCall({ run, onPicture }: { run: ImageRun; onPicture?: OnPicture }) {
  const made = run.made ?? [];
  const working = run.status === "switching" || run.status === "making" || run.status === "returning" || run.status === "queued";
  const share = run.progress ? progressShare(run.progress, run.count) : 0;
  // The prompt on one line, or all of it: the model's prompts run to paragraphs.
  const [whole, setWhole] = useState(false);
  return (
    <li className="monitor-kiln cut @container flex flex-col text-[#eae8e4] [--cut-size:14px]">
      <div className="flex min-w-0 items-start gap-2 px-3 py-2">
        <span className="grid h-5 shrink-0 place-items-center">
          <StatusMark status={markOf(run.status)} />
        </span>
        <button
          type="button"
          aria-expanded={whole}
          title={whole ? "Show the prompt on one line" : "Show the whole prompt"}
          onClick={() => setWhole((w) => !w)}
          className="flex min-w-0 flex-1 items-start gap-1.5 text-left text-sm hover:text-white"
        >
          <span className={`min-w-0 ${whole ? "whitespace-pre-wrap [overflow-wrap:anywhere]" : "truncate"}`}>{run.prompt || "make_image"}</span>
          <span className="grid h-5 shrink-0 place-items-center text-[#939ba4]">
            <IconChevron className={`transition-transform motion-reduce:transition-none ${whole ? "" : "rotate-180"}`} />
          </span>
        </button>
        <span className={`shrink-0 font-display text-xs leading-5 ${run.status === "failed" ? "text-[#ff7b6b]" : "text-[#939ba4]"}`}>
          {run.line ?? STATUS_WORDS[run.status]}
        </span>
      </div>
      {working && (
        <div className="h-0.5 bg-[#2e343c]" aria-hidden>
          <div className="h-full bg-[#ff5a1f] transition-[width] motion-reduce:transition-none" style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
      )}
      {(run.references.length > 0 || run.transparent || run.aspect) && (
        <p className="px-3 pb-2 text-xs text-[#939ba4]">
          {[run.references.length > 0 ? `From ${run.references.join(", ")}` : "", run.aspect ?? "", run.transparent ? "transparent" : ""]
            .filter(Boolean)
            .join(", ")}
        </p>
      )}
      {made.length > 0 && (
        // Side by side only when the reply itself is wide enough, whatever the window: the side columns narrow it.
        <ul className={`grid gap-px border-t border-[#2e343c] ${made.length > 1 ? "@2xl:grid-cols-2" : ""}`}>
          {made.map((picture) => (
            <Picture key={picture.name} picture={picture} onPicture={onPicture} />
          ))}
        </ul>
      )}
      {run.error && <p className="border-t border-[#2e343c] px-3 py-2 text-sm text-[#ff7b6b] [overflow-wrap:anywhere]">{run.error}</p>}
    </li>
  );
}

function Picture({ picture, onPicture }: { picture: MadePicture; onPicture?: OnPicture }) {
  if (picture.file) return <WorkspacePicture picture={picture} file={picture.file} onPicture={onPicture} />;
  return <Shown picture={picture} url={picture.url ?? null} onPicture={onPicture} />;
}

/** A picture that landed in the workspace: the version the call made, by its bytes in the page's store. */
function WorkspacePicture({ picture, file, onPicture }: { picture: MadePicture; file: Written; onPicture?: OnPicture }) {
  const found = useWorkspaceFile(file.workspace, file.file);
  const bytes = versionOf(found, file.version)?.bytes;
  const id = bytes?.store === "image" ? bytes.id : undefined;
  const lookup = usePictureUrl();
  const url = id ? lookup(id, "full") : null;
  const gone = found === null || (found !== undefined && id === undefined);
  return <Shown picture={picture} url={url} id={id} gone={gone ? `${file.name} is no longer in the workspace.` : undefined} onPicture={onPicture} />;
}

// Marking and the Image view sit over the picture, as on the user's own
// (PictureTools), and a click on it opens it full screen; the line under it
// wraps a whole piece at a time, so two pictures side by side never break a
// name or a seed.
function Shown({ picture, url, id, gone, onPicture }: { picture: MadePicture; url: string | null; id?: string; gone?: string; onPicture?: OnPicture }) {
  const chat: ChatPicture | null = url ? { name: `${picture.name}.png`, url, ...(id ? { id } : {}) } : null;
  const alt = picture.revisedPrompt ?? picture.name;
  const [full, setFull] = useState(false);
  return (
    <li className="flex min-w-0 flex-col">
      <div className="flex justify-center bg-[#23282f] p-2">
        {gone ? (
          <p className="px-1 py-6 text-sm text-[#939ba4]">{gone}</p>
        ) : (
          <span className="group/pic relative block max-w-full">
            <button type="button" aria-label={`Open ${picture.name} full screen`} title="Full screen" className="block max-w-full cursor-zoom-in" onClick={() => setFull(true)}>
              {url ? (
                <img src={url} alt={alt} className="stage-checker block max-h-[28rem] max-w-full object-contain" />
              ) : (
                <span role="img" aria-label={alt} className="block h-48 w-64 max-w-full bg-[#2e343c]" />
              )}
            </button>
            {onPicture && chat && <PictureTools picture={chat} onPicture={onPicture} />}
          </span>
        )}
      </div>
      {full && url && (
        <Lightbox onClose={() => setFull(false)}>
          <img src={url} alt={alt} className="stage-checker block max-h-[calc(100dvh-2rem)] max-w-[calc(100vw-2rem)] object-contain" />
        </Lightbox>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1.5 text-xs whitespace-nowrap text-[#939ba4] tabular-nums">
        <span className="font-display font-semibold text-[#eae8e4]">{picture.name}</span>
        <span>{sidesOf(picture.size)}</span>
        {picture.seed !== undefined && <span>Seed {picture.seed}</span>}
        {picture.kept && <span>Marked area only</span>}
        {picture.file && <span title="Kept as a file of the chat's workspace">{picture.file.name} in the workspace</span>}
        {picture.unkept && <span className="whitespace-normal text-[#ff7b6b]">Not saved in the workspace, kept in this reply: {picture.unkept}</span>}
        {url && (
          <button
            type="button"
            className="ml-auto flex items-center gap-1.5 font-display font-medium text-[#b9bec4] hover:text-white"
            onClick={() =>
              downloadImage({ id: 0, url, prompt: picture.name, size: picture.size, edit: false, ...(picture.seed !== undefined ? { seed: picture.seed } : {}) })
            }
          >
            <IconDownload />
            PNG
          </button>
        )}
      </div>
    </li>
  );
}
