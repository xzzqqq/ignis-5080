import { type ClipboardEvent, type KeyboardEvent, type ReactNode, useRef } from "react";
import { createPortal } from "react-dom";
import { formatBytes } from "../ui/bytes.ts";
import type { ContextUsage } from "../metrics/context.ts";
import { ContextMeter } from "../metrics/ContextMeter.tsx";
import type { AttachedChip } from "../workspace/attach.ts";
import { useAnchoredMenu } from "../ui/anchoredMenu.ts";
import { IconBrush, IconClose, IconImage, IconPaperclip, IconPlus } from "../ui/icons.tsx";
import { dataUriBytes, type PromptImage } from "./images.ts";

/** The hint line's idle words (spec speech/04, "The hint line"). */
export const IDLE_HINT = "Enter sends, Shift+Enter adds a line.";

/**
 * The prompt box under the conversation: Send, or Stop while this session
 * streams, the mic beside it (spec speech/04), the files this prompt adds to
 * the chat's workspace, the images it will carry, the model the next message
 * goes to, and the context bar.
 */
export function Composer(props: {
  value: string;
  onChange: (value: string) => void;
  /** A model can take the next message: the one loaded, or the one it switches to first. */
  ready: boolean;
  /** The picker of the model the next message goes to. */
  modelPicker?: ReactNode;
  /** A reply streams in some session. */
  busy: boolean;
  /** A reply streams in this session. */
  streamingHere: boolean;
  usage: ContextUsage;
  onSend: () => void;
  onStop: () => void;
  /** The Workspace switch is on: files can be attached, and the attached ones show. Pictures can be either way. */
  canAttach: boolean;
  /** The files the next prompt adds to the chat's workspace. */
  attachments: AttachedChip[];
  attachError: string | null;
  onAttach: (files: File[]) => void;
  onDetach: (name: string) => void;
  /** The images this prompt will be sent with, in the order they go on the wire. */
  images: PromptImage[];
  imageError: string | null;
  onAddImages: (files: File[]) => void;
  onRemoveImage: (index: number) => void;
  /** Opens a picture of the box in the markup editor. */
  onMarkImage?: (index: number) => void;
  /** Dictation into the prompt; its text lands at the cursor and is never sent by itself (Q4). */
}) {
  const { busy, streamingHere } = props;
  const hint = busy && !streamingHere ? "A reply is streaming in another session. Send works again once it ends." : IDLE_HINT;
  const picker = useRef<HTMLInputElement>(null);
  const imagePicker = useRef<HTMLInputElement>(null);
  /** An empty prompt still goes out when it carries an image or a file: the picture or the file is the question. */
  const hasPrompt = props.value.trim() !== "" || props.images.length > 0 || (props.canAttach && props.attachments.length > 0);

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      props.onSend();
    }
  }

  /** A pasted screenshot becomes an image of the prompt; pasted text is left to the textarea. */
  function onPaste(e: ClipboardEvent<HTMLTextAreaElement>) {
    const files = Array.from(e.clipboardData.files).filter((f) => f.type.startsWith("image/"));
    if (files.length === 0) return;
    e.preventDefault();
    props.onAddImages(files);
  }

  return (
    <div className="shrink-0 px-4 pb-4 md:px-10">
      {props.canAttach && (props.attachments.length > 0 || props.attachError) && (
        <div className="mx-auto mb-2 flex w-full max-w-3xl flex-wrap items-center gap-2">
          {props.attachments.map((a) => (
            <span key={a.name} className="cut flex min-w-0 items-center gap-1.5 bg-surface py-1 pr-1.5 pl-2.5 text-xs [--cut-size:6px]">
              <span className="min-w-0 truncate font-display font-semibold text-ink">{a.name}</span>
              <span className="shrink-0 font-display tabular-nums text-ash">
                {a.kind === "audio" ? `audio, ${formatBytes(a.size)}` : `${a.kind === "pdf" ? "PDF, " : ""}${a.characters.toLocaleString()} chars`}
              </span>
              <button type="button" aria-label={`Remove ${a.name}`} onClick={() => props.onDetach(a.name)} className="shrink-0 text-ash hover:text-fault">
                <IconClose />
              </button>
            </span>
          ))}
          {props.attachError && <span className="text-xs leading-snug text-fault">{props.attachError}</span>}
        </div>
      )}
      {(props.images.length > 0 || props.imageError) && (
        <div className="mx-auto mb-2 flex w-full max-w-3xl flex-wrap items-end gap-2">
          {props.images.map((image, index) => (
            <span key={`${image.name}:${index}`} className="cut relative block bg-surface p-1 [--cut-size:8px]">
              <img src={image.url} alt={image.name} className="block h-20 w-auto max-w-40 object-cover" />
              <span className="mt-1 block max-w-40 truncate text-center font-display text-[11px] tabular-nums text-ash">
                {image.width}×{image.height}, {Math.round(dataUriBytes(image.url) / 1024).toLocaleString()} KB
              </span>
              <button
                type="button"
                aria-label={`Remove ${image.name}`}
                onClick={() => props.onRemoveImage(index)}
                className="cut absolute top-1 right-1 grid size-6 place-items-center bg-surface text-ash [--cut-size:4px] hover:text-fault"
              >
                <IconClose />
              </button>
              {props.onMarkImage && (
                <button
                  type="button"
                  aria-label={`${image.markup ? "Change the marks on" : "Mark"} ${image.name}`}
                  title="Mark the area to change: the model sees the marks"
                  onClick={() => props.onMarkImage?.(index)}
                  className={`cut absolute top-1 left-1 grid size-6 place-items-center bg-surface [--cut-size:4px] hover:text-ink ${image.markup ? "text-ember" : "text-ash"}`}
                >
                  <IconBrush />
                </button>
              )}
            </span>
          ))}
          {props.imageError && <span className="pb-2 text-xs leading-snug text-fault">{props.imageError}</span>}
        </div>
      )}
      <div
        className={`cut relative mx-auto flex w-full max-w-3xl items-end gap-2 bg-surface p-2 [--cut-size:14px] shadow-[inset_0_-2px_0_var(--line)] focus-within:shadow-[inset_0_-2px_0_var(--ember)]`}
      >
        <AddMenu>
          {props.canAttach && (
            <AddMenuItem
              icon={<IconPaperclip />}
              label="File"
              hint="Text, PDF or audio, into the chat's workspace"
              onChoose={() => picker.current?.click()}
            />
          )}
          <AddMenuItem
            icon={<IconImage />}
            label="Image"
            hint="Sent with this prompt; needs a vision load"
            onChoose={() => imagePicker.current?.click()}
          />
        </AddMenu>
        {props.canAttach && (
          <input
            ref={picker}
            type="file"
            name="attachments"
            multiple
            hidden
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              e.target.value = "";
              if (files.length > 0) props.onAttach(files);
            }}
          />
        )}
        <input
          ref={imagePicker}
          type="file"
          name="images"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = "";
            if (files.length > 0) props.onAddImages(files);
          }}
        />
        <textarea
          className="max-h-60 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-normal text-ink [field-sizing:content] placeholder:text-ash focus:outline-none"
          rows={1}
          name="prompt"
          aria-label="Prompt"
          value={props.value}
          placeholder={props.ready ? "Ask anything" : "Waiting for the model…"}
          onChange={(e) => props.onChange(e.target.value)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
        />
        {streamingHere ? (
          <button
            type="button"
            className="cut bg-[#c8161d] px-5 py-2.5 font-display text-sm font-semibold text-white hover:bg-[#a8121a]"
            onClick={props.onStop}
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="cut bg-[#ff5a1f] px-5 py-2.5 font-display text-sm font-semibold text-[#1c2026] hover:bg-[#ff7a45] disabled:cursor-default disabled:bg-line disabled:text-ash"
            disabled={busy || !hasPrompt || !props.ready}
            onClick={props.onSend}
          >
            Send
          </button>
        )}
      </div>
      <div className="mx-auto mt-2 flex w-full max-w-3xl items-center justify-between gap-4 px-1">
        <div className="flex min-w-0 items-center gap-4">
          {props.modelPicker}
          <p className="min-w-0 text-xs text-ash">
            {hint}
          </p>
        </div>
        <ContextMeter usage={props.usage} />
      </div>
    </div>
  );
}

/**
 * The `+` beside the prompt and what it offers: one button for everything a
 * prompt can be given, rather than an icon per kind growing along the box.
 * It opens upwards — the composer sits at the foot of the page — and closes
 * on a choice, on Escape, or on a click anywhere else.
 */
function AddMenu({ children }: { children: ReactNode }) {
  // A portal placed by hand: the prompt box is a `.cut` (useAnchoredMenu says why).
  const menu = useAnchoredMenu();
  return (
    <div className="shrink-0">
      <button
        ref={menu.button}
        type="button"
        aria-label="Add to this prompt"
        title="Add a file or an image"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        className={`grid size-10 place-items-center hover:text-ink [&>svg]:size-[18px] ${menu.open ? "text-ink" : "text-ash"}`}
        onClick={menu.toggle}
      >
        <IconPlus />
      </button>
      {menu.open &&
        createPortal(
          // The click that chose an item is what closes the menu, so the choice
          // is handled on the way out and never lands on a dismissed menu.
          <div
            ref={menu.menu}
            role="menu"
            style={{ left: menu.at.left, bottom: menu.at.bottom }}
            className="cut fixed z-40 flex w-56 flex-col bg-surface py-1 shadow-[0_8px_24px_rgba(28,32,38,0.28)] [--cut-size:10px]"
            onClick={menu.close}
          >
            {children}
          </div>,
          document.body,
        )}
    </div>
  );
}

/** One thing the `+` can add: what it is, and a line on what happens to it. */
function AddMenuItem({
  icon,
  label,
  hint,
  onChoose,
}: {
  icon: ReactNode;
  label: string;
  hint: string;
  onChoose: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      className="flex w-full items-start gap-2.5 px-3 py-2 text-left text-ash hover:bg-line hover:text-ink"
      onClick={onChoose}
    >
      <span className="mt-0.5 shrink-0">{icon}</span>
      <span className="min-w-0">
        <span className="block font-display text-sm font-semibold text-ink">{label}</span>
        <span className="block text-xs leading-snug">{hint}</span>
      </span>
    </button>
  );
}
