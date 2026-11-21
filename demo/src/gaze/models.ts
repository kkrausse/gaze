// The models the demo offers. Files are fetched from Hugging Face (the only network requests the page makes):
// <HF><repo>/resolve/main/<file>. The prefix can be replaced (a mirror with the same layout) by
// <meta name="model-base" content="..."> in index.html (empty = Hugging Face) or ?hf=<url>.

export interface ModelEntry {
  id: string; // short id, shown in the picker
  repo: string; // Hugging Face model repo
  label: string;
  available: boolean; // false: listed, not published yet
  licence: string; // of the weights
}

export const HF = "https://huggingface.co/";

export const MODELS: ModelEntry[] = [
  { id: "m68", repo: "kkrausse/gaze-nc", label: "m68 (gaze-nc)", available: true, licence: "Sapiens2 License, non-commercial use only" },
  { id: "m69", repo: "kkrausse/gaze", label: "m69 (gaze)", available: true, licence: "Sapiens2 License" },
];

export function hfBase(): string {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="model-base"]')?.content.trim();
  const o = new URLSearchParams(location.search).get("hf") || meta;
  return o ? (o.endsWith("/") ? o : o + "/") : HF;
}

export const fileUrl = (m: ModelEntry, file: string) => `${hfBase()}${m.repo}/resolve/main/${file}`;

/** ?model=<id> picks the model (else the first available). */
export function initialModel(): ModelEntry {
  const id = new URLSearchParams(location.search).get("model");
  return MODELS.find((m) => m.id === id && m.available) ?? MODELS.find((m) => m.available) ?? MODELS[0];
}
