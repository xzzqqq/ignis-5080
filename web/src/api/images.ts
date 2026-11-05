// Serialized progress kept only to render pictures in pre-existing chat sessions.
export type ImagePhase = "switching" | "loading" | "enhancing" | "encoding" | "denoising" | "decoding";
export type ImageProgress = { phase: ImagePhase; image: number; step: number; total: number };
