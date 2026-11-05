import { useCallback, useEffect, useState } from "react";
import { forgetKey, useAuth } from "../api/auth.ts";
import { canChat, canSend, loadUnderWay, type ModelState, useModel } from "../api/model.ts";
import { Composer } from "../conversation/Composer.tsx";
import { type ChatPicture, chatImageFromFile, type PictureAction, type PromptImage } from "../conversation/images.ts";
import { useNextModel } from "../conversation/useNextModel.ts";
import { type OpenAgent, Transcript } from "../conversation/Transcript.tsx";
import { contextUsage } from "../metrics/context.ts";
import { Readout } from "../metrics/Readout.tsx";
import { MonitorView } from "../monitor/MonitorView.tsx";
import { useMonitor, useMonitorVisible } from "../monitor/useMonitor.ts";
import { SessionLog } from "../sessions/SessionLog.tsx";
import { SessionsPanel } from "../sessions/SessionsPanel.tsx";
import { StorePictures } from "../store/pictureViews.tsx";
import { fileFromDataUrl } from "../store/pictureFiles.ts";
import type { Mark, MarkedPicture } from "../markup/markup.ts";
import { MarkupEditor } from "../markup/MarkupEditor.tsx";
import { ServerView } from "../server/ServerView.tsx";
import { useServer, useTick } from "../server/useServer.ts";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "../settings/defaults.ts";
import { escapeFree } from "../ui/Sheet.tsx";
import { pageStore } from "../store/index.ts";
import { SettingsPanel } from "../settings/SettingsPanel.tsx";
import { AgentReader } from "../tools/agents/AgentReader.tsx";
import { MemoryManager } from "../tools/local/MemoryManager.tsx";
import { AGENT_SYSTEM_PROMPT, type AgentRun } from "../tools/agents/agents.ts";
import { ALL_TOOLS, type ToolsState } from "../tools/index.ts";
import { refusal } from "../workspace/files.ts";
import { useWorkspace, useWorkspaceList } from "../workspace/useWorkspaces.ts";
import { StoreWorkspaceFiles } from "../workspace/WorkspaceFiles.tsx";
import { WorkspacePanel } from "../workspace/WorkspacePanel.tsx";
import { ChatWorkspaceBar } from "../workspace/WorkspacePicker.tsx";
import { pageWorkspaces } from "../workspace/workspaces.ts";
import { Header, type View } from "./Header.tsx";
import { KeyPage } from "./KeyPage.tsx";
import { TransportNotice } from "./TransportNotice.tsx";
import { STORED_FLAGS, useStoredFlag } from "./storedFlag.ts";
import { useConversation } from "./useConversation.ts";

// The Playground (GitHub #164): a streaming chat against ignis — over its
// Responses WebSocket, or /v1/chat/completions (GitHub #283) — with
// per-request figures measured in the browser.
// The chat's, the Decide tab's and the Image view's sessions are kept in this
// browser's store and come back on a reload; settings live in memory, apart
// from the switches this browser stores (`storedFlag.ts`).
// This is the page's layout; the conversation loop is useConversation.
// The header switches between the chat, the Decide tab (GitHub #247), —
// when ignis serves metrics — the Monitor (GitHub #165), the Server view
// (GitHub #316), the Image view and the Transcribe view (spec speech/07).
// Whichever is showing, the chat stays mounted underneath, so a streaming
// reply carries on, and a listening goes on whichever view shows.

type Drawer = "sessions" | "settings" | null;

/** A picture of the chat open in the markup editor, and what saving the marks does with them. */
type ChatMarking = { url: string; title: string; marks?: Mark[]; keepOutside?: boolean; apply: (marked: MarkedPicture | null) => Promise<void> };

/** A picture of the chat at its own size: the one kept beside the redraw, else the redraw itself. */
const blobOf = (picture: { full?: Blob; url: string }, name: string): Blob => picture.full ?? fileFromDataUrl(picture.url, name);

const stem = (name: string) => name.replace(/\.[^.]+$/, "");

export function App() {
  const auth = useAuth();
  const [model, refreshModel] = useModel();
  const monitorVisible = useMonitorVisible();
  const [view, setView] = useState<View>("chat");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tools, setTools] = useState<ToolsState>(ALL_TOOLS);
  const [markdown, setMarkdown] = useState(true);
  // Off: the page runs one turn at a time, whichever session it is in. Stored,
  // so turning it on survives a reload rather than looking like it never took.
  const [parallel, setParallel] = useStoredFlag(STORED_FLAGS.parallel, false);
  const [input, setInput] = useState("");
  // The images the next prompt will carry. They belong to the box, not to a
  // session: once sent they live on the user message, and the box is empty again.
  const [images, setImages] = useState<PromptImage[]>([]);
  const [imageError, setImageError] = useState<string | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [reader, setReader] = useState<OpenAgent | null>(null);
  const closeReader = useCallback(() => setReader(null), []);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const closeMemory = useCallback(() => setMemoryOpen(false), []);
  const [drawer, setDrawer] = useState<Drawer>(null);
  // The chat follows the model served by this process.
  const nextModel = useNextModel(model);
  const chat = useConversation({ model, next: nextModel.next, refreshModel, settings, tools, parallel });
  const { active } = chat;
  // The chat's workspace (spec #170): its picker in the chat's header, and the panel of its files on the right.
  const workspaces = pageWorkspaces();
  const chatWorkspace = useWorkspace(workspaces, active.workspaceId);
  const workspaceList = useWorkspaceList(workspaces);
  const [workspaceOpen, setWorkspaceOpen] = useStoredFlag(STORED_FLAGS.workspacePanel, false);
  const [workspaceNotice, setWorkspaceNotice] = useState<string | null>(null);
  const monitoring = view === "monitor" && monitorVisible;
  const serving = view === "server";
  const [serverOpened, setServerOpened] = useState(false);
  const [chatMarking, setChatMarking] = useState<ChatMarking | null>(null);
  // A switch or a load runs the header's flame and heat bar as a reply does (spec playground/05).
  const busy = chat.busy || loadUnderWay(model);

  const readerRun: AgentRun | undefined = reader
    ? active.messages.find((m) => m.id === reader.messageId)?.agents?.find((r) => r.callId === reader.callId)
    : undefined;

  // Metrics going away drops back to the chat *from the Monitor*, so their
  // return never flips the page on its own — and a reader in the Decide tab,
  // which needs no metrics, is left where they are.
  useEffect(() => {
    if (!monitorVisible) setView((current) => (current === "monitor" ? "chat" : current));
  }, [monitorVisible]);

  useEffect(() => {
    if (!drawer) return;
    // An Esc a full screen took closes that alone.
    const close = (e: globalThis.KeyboardEvent) => escapeFree(e) && setDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer]);

  function selectSession(id: number) {
    chat.selectSession(id);
    setDrawer(null);
    setReader(null);
  }

  function newSession() {
    chat.newSession();
    setDrawer(null);
  }

  function send() {
    const text = input.trim();
    // A dictation under way is never sent: Send waits until it ends (Q4). Files attached are a prompt by themselves.
    const files = tools.enabled && tools.workspace ? chat.attachments.length : 0;
    if ((!text && images.length === 0 && files === 0) || !chat.canRun) return;
    setInput("");
    setImages([]);
    setImageError(null);
    // A prompt whose files the store refused is not sent: the box gets back what it held, ahead of anything typed since.
    void chat.send(text, images).then((sent) => {
      if (sent) return;
      setInput((now) => [text, now].filter((t) => t !== "").join("\n"));
      setImages((now) => [...images, ...now]);
    });
  }

  /** Picked or pasted files as prompt images; the ones that cannot be read are named instead. */
  async function addImages(files: File[]) {
    // The chat keeps each picture at its own size too: the picture tool edits that one.
    const results = await Promise.all(files.map((file) => chatImageFromFile(file)));
    const errors = results.flatMap((r) => (r.ok ? [] : [r.error]));
    setImageError(errors.length > 0 ? errors.join(" ") : null);
    const added = results.flatMap((r) => (r.ok ? [r.image] : []));
    if (added.length > 0) setImages((current) => [...current, ...added]);
  }

  /** The marked picture as a prompt image of the chat: the drawn version on the wire, the marks and the original beside. */
  async function markedImage(source: Blob, name: string, marked: MarkedPicture): Promise<PromptImage | null> {
    const read = await chatImageFromFile(marked.drawn, `${stem(name)}-marked.png`);
    if (!read.ok) {
      setImageError(read.error);
      return null;
    }
    return { ...read.image, markup: { source, marks: marked.marks, keepOutside: marked.keepOutside, region: marked.region } };
  }

  /** Opens `source` in the markup editor; `apply` gets what was saved. */
  function openMarking(source: Blob, title: string, apply: ChatMarking["apply"], earlier?: PromptImage["markup"]) {
    setChatMarking({
      url: URL.createObjectURL(source),
      title,
      ...(earlier ? { marks: earlier.marks, keepOutside: earlier.keepOutside } : {}),
      apply,
    });
  }

  function closeMarking() {
    if (chatMarking) URL.revokeObjectURL(chatMarking.url);
    setChatMarking(null);
  }

  /** A picture of the prompt box marked again from its original; it keeps its place in the box. */
  function markBoxImage(index: number) {
    const image = images[index];
    if (!image) return;
    const source = image.markup?.source ?? blobOf(image, image.name);
    openMarking(
      source,
      `Mark ${image.name}`,
      async (marked) => {
        const next = marked ? await markedImage(source, image.name, marked) : image.markup ? await clean(source, image.name) : null;
        if (next) setImages((current) => current.map((x) => (x === image ? next : x)));
      },
      image.markup,
    );
  }

  async function clean(source: Blob, name: string): Promise<PromptImage | null> {
    const read = await chatImageFromFile(source, name);
    if (read.ok) return read.image;
    setImageError(read.error);
    return null;
  }

  /** A picture of the conversation: marked into the prompt box, or opened in a new session of the Image view. */
  async function onPicture(picture: ChatPicture, _action: PictureAction) {
    // One the page's store holds (a workspace's) is read from there; the store's bytes are what the Image view keeps.
    const stored = picture.id && !picture.full ? await pageStore().getImage(picture.id).catch(() => null) : null;
    if (picture.id && !picture.full && !stored) return;
    const source = stored ?? blobOf(picture, picture.name);
    openMarking(source, `Mark ${picture.name}`, async (marked) => {
      const image = marked && (await markedImage(source, picture.name, marked));
      if (image) setImages((current) => [...current, image]);
    });
  }

  const set = <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  // A 401 anywhere shows the key prompt in place of the page; App stays
  // mounted, so the sessions are still here once the key is in.
  if (auth.needsKey) return <KeyPage rejected={auth.rejected} />;

  return (
    // Stored pictures are shown by URLs read once from the page's store: the Image view's, and the chat's. A reply's
    // file card reads the version it wrote from the chat's workspace.
    <StorePictures store={pageStore()}>
    <StoreWorkspaceFiles workspaces={pageWorkspaces()}>
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header
        model={model}
        busy={busy}
        onOpen={setDrawer}
        onForgetKey={auth.key ? forgetKey : undefined}
        onOpenMemory={() => setMemoryOpen(true)}
        view={monitoring ? "monitor" : serving ? "server" : "chat"}
        onView={(next) => {
          if (next === "server") setServerOpened(true);
          setView(next);
        }}
        monitorAvailable={monitorVisible}
      />

      <TransportNotice />

      {monitoring && <Monitor />}

      {serverOpened && (
        <div className={serving ? "contents" : "hidden"}>
          <Server model={model} refreshModel={refreshModel} streaming={chat.busy} visible={serving} />
        </div>
      )}

      <div className={monitoring || serving ? "hidden" : "contents"}>
        <div className="relative flex min-h-0 flex-1">
          {drawer && (
            <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => setDrawer(null)} aria-hidden />
          )}

          <SessionsPanel
            open={drawer === "sessions"}
            list={chat.list}
            activeId={active.id}
            streaming={chat.streaming}
            onNew={newSession}
            onSelect={selectSession}
            onRemove={chat.deleteSession}
            notice={chat.storeError}
          />

          <main className="flex min-w-0 flex-1 flex-col">
            <ChatWorkspaceBar
              workspace={chatWorkspace.workspace}
              files={chatWorkspace.files}
              list={workspaceList}
              deleted={active.workspaceId !== undefined && chatWorkspace.workspace === null}
              held={chat.workspaceHeld}
              onPick={(id) => setWorkspaceNotice(chat.pickWorkspace(id, chatWorkspace.workspace?.name))}
              onRename={async (name) => {
                const id = chatWorkspace.workspace?.id;
                if (!id) return null;
                try {
                  const renamed = await workspaces.rename(id, name);
                  return renamed.ok ? null : renamed.error;
                } catch (err) {
                  return refusal(err);
                }
              }}
              onDelete={() => {
                const id = chatWorkspace.workspace?.id;
                setWorkspaceNotice(null);
                if (id) void chat.deleteWorkspace(id).catch((err) => setWorkspaceNotice(refusal(err)));
              }}
              open={workspaceOpen}
              onToggle={() => setWorkspaceOpen(!workspaceOpen)}
              notice={workspaceNotice}
            />
            <Transcript
              session={active}
              model={model}
              markdown={markdown}
              canRerun={chat.canRun}
              following={chat.following}
              openAgent={reader}
              onOpenAgent={setReader}
              onRerun={chat.rerun}
              onSave={chat.saveEdit}
              onFork={chat.fork}
              onAnswer={chat.answer}
              onPicture={(picture, action) => void onPicture(picture, action)}
            />
            <Composer
              value={input}
              onChange={setInput}
              ready={canSend(model) && nextModel.next.id !== null}
              busy={chat.sendBlocked}
              streamingHere={chat.streamingHere}
              usage={contextUsage(active.log, settings.maxTokens, canChat(model) ? model.contextLimit : null)}
              onSend={send}
              onStop={chat.stop}
              canAttach={tools.enabled && tools.workspace}
              attachments={chat.attachments}
              attachError={chat.attachError}
              onAttach={(files) => void chat.attach(files)}
              onDetach={chat.detach}
              images={images}
              imageError={imageError}
              onAddImages={(files) => void addImages(files)}
              onRemoveImage={(index) => setImages((current) => current.filter((_, i) => i !== index))}
              onMarkImage={markBoxImage}
            />
          </main>

          {workspaceOpen && <WorkspacePanel workspaces={workspaces} view={chatWorkspace} chat={chat.binding} chats={chat.chatNames} onClose={() => setWorkspaceOpen(false)} />}

          <SettingsPanel
            open={drawer === "settings"}
            settings={settings}
            set={set}
            tools={tools}
            onToolsChange={setTools}
            markdown={markdown}
            onMarkdownChange={setMarkdown}
            parallel={parallel}
            onParallelChange={setParallel}
            onOpenMemory={() => setMemoryOpen(true)}
          />
        </div>

        <SessionLog rows={active.log} open={logOpen} onToggle={() => setLogOpen((o) => !o)} />
      </div>

      {memoryOpen && <MemoryManager onClose={closeMemory} />}

      {chatMarking && (
        <MarkupEditor
          source={{ url: chatMarking.url, ...(chatMarking.marks ? { marks: chatMarking.marks, mode: "drawn" as const, keepOutside: chatMarking.keepOutside } : {}) }}
          title={chatMarking.title}
          modes={["drawn"]}
          onSave={(marked) => {
            const apply = chatMarking.apply;
            closeMarking();
            void apply(marked);
          }}
          onClose={closeMarking}
        />
      )}

      {readerRun && (
        <>
          <div className="fixed inset-0 z-40 bg-[#1c2026]/40" onClick={closeReader} aria-hidden />
          <AgentReader
            key={readerRun.callId}
            run={readerRun}
            markdown={markdown}
            systemPrompt={readerRun.systemPrompt ?? AGENT_SYSTEM_PROMPT}
            figures={readerRun.figures && <Readout figures={readerRun.figures} />}
            onClose={closeReader}
          />
        </>
      )}
    </div>
    </StoreWorkspaceFiles>
    </StorePictures>
  );
}

/** The Monitor subscribes on its own, so a scrape re-renders it and not the chat. */
function Monitor() {
  return <MonitorView state={useMonitor()} />;
}

/** The Server view's state lives here, and its clock ticks only while a switch runs. */
function Server({ model, refreshModel, streaming, visible }: { model: ModelState; refreshModel: () => void; streaming: boolean; visible: boolean }) {
  const [state, actions] = useServer(model, refreshModel, visible);
  const now = useTick(loadUnderWay(model));
  return <ServerView model={model} state={state} actions={actions} streaming={streaming} now={now} />;
}

