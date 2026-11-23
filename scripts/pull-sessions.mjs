// Download every stored session to ./sessions/. Usage (token from Vercel > Storage > taste-sessions > .env.local):
//   BLOB_READ_WRITE_TOKEN=... node scripts/pull-sessions.mjs
// Without local Node: docker run --rm -e BLOB_READ_WRITE_TOKEN -v "$PWD":/w -w /w node:22-alpine \
//   sh -c "npm i --no-save @vercel/blob >/dev/null && node scripts/pull-sessions.mjs"
import { list, get } from "@vercel/blob";
import { mkdir, writeFile } from "node:fs/promises";

await mkdir("sessions", { recursive: true });
let cursor, n = 0;
do {
  const page = await list({ prefix: "sessions/", cursor, limit: 1000 });
  for (const b of page.blobs) {
    const r = await get(b.pathname, { access: "private" });
    await writeFile(b.pathname, Buffer.from(await new Response(r.stream).arrayBuffer()));
    n++;
  }
  cursor = page.cursor;
} while (cursor);
console.log(`saved ${n} sessions to ./sessions/`);
