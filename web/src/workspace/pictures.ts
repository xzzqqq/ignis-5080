import type { PromptImage } from "../conversation/images.ts";
import { fileFromDataUrl } from "../store/pictureFiles.ts";
import type { AttachedFile } from "./attach.ts";
import type { WorkspaceFile } from "./files.ts";
import type { ChatBinding, Workspaces, Written } from "./workspaces.ts";

// Pictures as workspace files (#394): the pictures make_image makes land in
// the chat's workspace, born for them if needed, and the reply shows them from
// there; a picture the user attaches or pastes reaches the model inline in its
// turn, as it always did, and becomes a file of the workspace too. Their
// bytes are kept once, in the image store image sessions use.

/** Where make_image's pictures go, and where a later call that edits one reads it back. */
export type MadeLanding = {
  /** A picture made for the chat, as a file of its workspace authored by the chat, which has seen it; rejects when the store refuses it. */
  land: (picture: Blob, name: string) => Promise<Written>;
  /** The bytes of the version a picture landed as; null once its workspace is gone. */
  read: (file: Written) => Promise<Blob | null>;
};

export function madeIn(workspaces: Workspaces, chat: ChatBinding): MadeLanding {
  return {
    async land(picture, name) {
      const [written] = await workspaces.add(await chat.workspace(), [{ name, kind: "picture", blob: picture }], chat.reader);
      if (!written) throw new Error("the workspace did not take the picture");
      return written;
    },
    async read(file) {
      const version = versionOf(await workspaces.file(file.workspace, file.file), file.version);
      return version ? workspaces.content(version) : null;
    },
  };
}

/** Version `n` of `file`, when the file is there and has it. */
export const versionOf = (file: WorkspaceFile | null | undefined, n: number) => file?.versions.find((v) => v.n === n);

/** The pictures of the prompt box as the files the workspace keeps: each as it was picked (or as marked), not the redraw for the wire. */
export function boxPictures(images: PromptImage[]): AttachedFile[] {
  return images.map((image) => {
    const blob = image.full ?? fileFromDataUrl(image.url, image.name);
    return { name: image.name, kind: "picture", blob, size: blob.size };
  });
}
