import { createContext, type ReactNode, useContext, useMemo, useReducer, useRef } from "react";
import type { ImageId, ImageStore } from "../store/imageStore.ts";

// How the image view shows a stored picture: an object URL for its bytes or
// its thumbnail, made once per picture and kept for the page's life. Reading
// is asynchronous, so the first ask answers null and the view renders again
// once the URL is there. Tests hand in a plain lookup instead.

export type PictureKind = "full" | "thumb";

/** The URL to show a stored picture by, or null while it is being read (or is gone). */
export type PictureUrl = (id: ImageId, kind?: PictureKind) => string | null;

const Pictures = createContext<PictureUrl>(() => null);

export const usePictureUrl = () => useContext(Pictures);

/** Lookups from a plain map: what the view's tests give it. */
export function StaticPictures({ urls, children }: { urls: Record<string, string>; children: ReactNode }) {
  const lookup: PictureUrl = (id, kind = "full") => urls[`${kind}:${id}`] ?? urls[id] ?? null;
  return <Pictures.Provider value={lookup}>{children}</Pictures.Provider>;
}

/** Lookups read from `store`, each picture once. */
export function StorePictures({ store, children }: { store: ImageStore; children: ReactNode }) {
  const cache = useRef(new Map<string, string | "reading" | "missing">());
  const [version, landed] = useReducer((n: number) => n + 1, 0);
  const lookup = useMemo<PictureUrl>(() => {
    // `version` makes a new lookup, and so a new render of the readers, each time a URL lands.
    void version;
    return (id, kind = "full") => {
      const key = `${kind}:${id}`;
      const hit = cache.current.get(key);
      if (hit === undefined) {
        cache.current.set(key, "reading");
        const read = kind === "thumb" ? store.getThumb(id).then((thumb) => thumb ?? store.getImage(id)) : store.getImage(id);
        void read.then(
          (blob) => {
            cache.current.set(key, blob ? URL.createObjectURL(blob) : "missing");
            landed();
          },
          () => cache.current.set(key, "missing"),
        );
        return null;
      }
      return hit === "reading" || hit === "missing" ? null : hit;
    };
  }, [store, version]);
  return <Pictures.Provider value={lookup}>{children}</Pictures.Provider>;
}

/** A stored picture, or the quiet box it fills once read. */
export function Picture({ id, kind = "thumb", alt, className = "" }: { id: ImageId; kind?: PictureKind; alt: string; className?: string }) {
  const url = usePictureUrl()(id, kind);
  return url ? <img src={url} alt={alt} className={className} draggable={false} /> : <span role="img" aria-label={alt} className={`block bg-line/60 ${className}`} />;
}
