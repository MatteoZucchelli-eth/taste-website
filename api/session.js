// Stores one anonymous game session in the private Blob store connected to this project (BLOB_READ_WRITE_TOKEN).
// Only whitelisted fields are kept: no name, IP address, user agent or other device info is ever written.
import { put } from "@vercel/blob";

const MAX_BYTES = 300_000;
const KEYS = ["v", "id", "partial", "at", "model", "fromChallenge", "tag", "nTrain", "nTest", "nRandom", "mixRandom",
              "train", "test", "z", "sideBias", "hits", "topRate"];

export async function POST(request) {
  const text = await request.text();
  if (!text || text.length > MAX_BYTES) return new Response("bad size", { status: 413 });
  let rec;
  try { rec = JSON.parse(text); } catch { return new Response("bad json", { status: 400 }); }
  const id = String(rec?.id ?? "").replace(/[^\w-]/g, "").slice(0, 80);
  if (!id || !Array.isArray(rec.train)) return new Response("bad record", { status: 400 });
  const clean = Object.fromEntries(KEYS.filter((k) => k in rec).map((k) => [k, rec[k]]));
  clean.id = id;
  clean.receivedAt = new Date().toISOString().slice(0, 10); // day only
  // Same id = same session: progress checkpoints are overwritten by the final record.
  await put(`sessions/${id}.json`, JSON.stringify(clean), {
    access: "private", allowOverwrite: true, addRandomSuffix: false, contentType: "application/json",
  });
  return new Response(null, { status: 204 });
}
