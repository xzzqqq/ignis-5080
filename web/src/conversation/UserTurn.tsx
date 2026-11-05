import { memo, useState } from "react";
import type { Message, TurnSwitch } from "../sessions/sessions.ts";
import { IconBrush, IconFork, IconPencil } from "../ui/icons.tsx";
import type { ChatPicture, PictureAction } from "./images.ts";
import { MessageEditor } from "./MessageEditor.tsx";
import { ActionButton, ActionRow, EditedMark, RERUN_BLOCKED, sameTurnProps, type TurnActions } from "./TurnActions.tsx";

/** A prompt of the transcript; one that did not change is not rendered again. */
export const UserTurn = memo(Prompt, sameTurnProps);

function Prompt({
  message: m,
  actions,
  onResend,
  onPicture,
}: {
  message: Message;
  actions: TurnActions;
  onResend: (text: string) => void;
  onPicture?: (picture: ChatPicture, action: PictureAction) => void;
}) {
  const [editing, setEditing] = useState(false);
  if (editing) {
    return (
      <MessageEditor
        initial={m.content}
        onCancel={() => setEditing(false)}
        buttons={[
          {
            label: "Save",
            onClick: (text) => {
              actions.onSave(text);
              setEditing(false);
            },
          },
          {
            label: "Save and resend",
            primary: true,
            disabled: !actions.canRerun,
            title: actions.canRerun ? "Drops everything after this prompt and sends it again" : RERUN_BLOCKED,
            onClick: (text) => {
              setEditing(false);
              onResend(text);
            },
          },
        ]}
      />
    );
  }
  return (
    <div className="group/turn flex flex-col items-end gap-1 self-end md:max-w-[85%]">
      {m.images && m.images.length > 0 && (
        <div className="flex flex-wrap justify-end gap-2">
          {/* A download, not a new tab: Chrome refuses to navigate the top frame to a `data:` URL. */}
          {m.images.map((image, index) => (
            <span key={`${image.name}:${index}`} className="group/pic relative block">
              <a href={image.url} download={image.name} title={`Save ${image.name}`}>
                <img src={image.url} alt={image.name} className="cut block max-h-56 w-auto max-w-full [--cut-size:10px]" />
              </a>
              {onPicture && <PictureTools picture={image} onPicture={onPicture} />}
            </span>
          ))}
        </div>
      )}
      {m.content !== "" && (
        <article className="cut bg-surface px-4 py-3 [--cut-size:12px]">
          <pre className="whitespace-pre-wrap break-words font-sans text-[15px] leading-normal">{m.content}</pre>
        </article>
      )}
      {m.attached && m.attached.length > 0 && (
        <ul aria-label="Attached to the workspace" className="flex flex-wrap justify-end gap-1.5">
          {m.attached.map((name) => (
            <li key={name} className="cut bg-surface px-2.5 py-1 font-display text-xs font-semibold text-ink [--cut-size:6px]">
              {name}
            </li>
          ))}
        </ul>
      )}
      {m.switch && <SwitchLine change={m.switch} />}
      <ActionRow className="-mr-2">
        {m.edited && <EditedMark />}
        <ActionButton icon={<IconPencil />} label="Edit" onClick={() => setEditing(true)} />
        <ActionButton icon={<IconFork />} label="Fork" title="Start a new session from this message" onClick={actions.onFork} />
      </ActionRow>
    </div>
  );
}

/** The switch the turn made before its first request, under the prompt it answered. */
export function SwitchLine({ change }: { change: TurnSwitch }) {
  const from = change.from ? ` from ${change.from}` : "";
  const words = {
    switching: `Switching to ${change.to}${from}: the reply starts once it is loaded.`,
    switched: `Sent to ${change.to}, switched${from}.`,
    stopped: `Stopped while switching to ${change.to}. The switch finishes on the server.`,
    failed: `Not sent: the switch to ${change.to} did not land. ${change.error ?? ""}`.trim(),
  }[change.status];
  return (
    <p
      role={change.status === "failed" ? "alert" : "status"}
      className={`flex max-w-full items-center gap-2 font-display text-xs ${change.status === "failed" ? "text-fault" : change.status === "switching" ? "text-ember" : "text-ash"}`}
    >
      <span
        className={`cut size-2 shrink-0 [--cut-size:3px] ${change.status === "switching" ? "animate-pulse bg-ember motion-reduce:animate-none" : change.status === "failed" ? "bg-fault" : "bg-line"}`}
        aria-hidden
      />
      <span className="min-w-0 [overflow-wrap:anywhere]">{words}</span>
    </p>
  );
}

/** Over a picture of the conversation: mark it into the prompt box, or open it in the Image view. */
export function PictureTools({ picture, onPicture }: { picture: ChatPicture; onPicture: (picture: ChatPicture, action: PictureAction) => void }) {
  const tool = "grid size-7 place-items-center bg-[#1c2026]/80 text-[#eae8e4] hover:bg-[#ff5a1f] hover:text-[#1c2026]";
  return (
    <span className="absolute right-1.5 bottom-1.5 flex gap-1 opacity-0 transition-opacity group-focus-within/pic:opacity-100 group-hover/pic:opacity-100 motion-reduce:transition-none max-lg:opacity-100">
      <button type="button" aria-label={`Mark ${picture.name}`} title="Mark the area to change, into the prompt box" className={tool} onClick={() => onPicture(picture, "mark")}>
        <IconBrush />
      </button>

    </span>
  );
}
