"use strict";
/*
 * This or That: Phase C taste model in the browser.
 *   score s(x) = z . U(x)    (U = 9 precomputed axes per image; axis 0 ~ crowd consensus)
 *   P(top chosen) = sigmoid(s(top) - s(bottom) + b)   b = per-person side bias (top/bottom or left/right)
 *   prior z ~ N(mu, Sigma / tau), b ~ N(0, sideBiasSd^2); MAP by Newton; Laplace posterior for question selection.
 * Questions 1..nRandom are random pairs, the rest maximise expected information about z
 * (p(1-p) d^T H_z^-1 d, as in src/active.py). Test pairs (at random positions): nSure are the pair the model is most sure
 * about among nCand random candidates, nContra the most-sure pair on which the crowd model (z = mu) picks the other image, the rest
 * random (simulated in src/sure_test.py); the frozen model's guess is locked before the pair is shown. Positions are randomised on every pair, and b soaks up any side preference so it does not leak into z.
 */
const CFG = window.TASTE_CONFIG;
// URL overrides for long data-collection sessions, e.g. ?train=100&test=20&mix=1
//   train/test = number of learning / test pairs; mix=1 alternates random and infogain questions after the first nRandom
//   (random pairs keep the data representative for fitting other critics later; infogain sharpens z).
{
  const q = new URLSearchParams(location.search);
  const int = (k, lo, hi) => { const v = parseInt(q.get(k), 10); return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : null; };
  CFG.nTrain = int("train", 5, 400) ?? CFG.nTrain;
  CFG.nTest = int("test", 1, 100) ?? CFG.nTest;
  CFG.nSure = Math.min(CFG.nTest, int("sure", 0, 100) ?? CFG.nSure ?? 0);
  CFG.nContra = Math.min(CFG.nTest - CFG.nSure, int("contra", 0, 100) ?? CFG.nContra ?? 0);
  CFG.mixRandom = q.get("mix") === "1";
  CFG.tag = (q.get("tag") || "").replace(/[^\w-]/g, "").slice(0, 40);
}
/** Question n (0-based) is a random pair? */
const isRandomQ = (n) => n < CFG.nRandom || (CFG.mixRandom && (n - CFG.nRandom) % 2 === 1);
const $ = (id) => document.getElementById(id);
/** Image URL: the local server mirrors the dataset paths; the Vercel build renames images to <index>.jpg. */
/** Anonymous funnel event for Vercel Web Analytics (no-op locally; custom events show up on Vercel Pro and above). */
const track = (name, data) => { try { window.va && window.va("event", { name, data }); } catch {} };
const imgUrl = (i) => CFG.imgBase + (CFG.imgByIndex ? `${i}.jpg` : M.path[i]);

// Pole names, read off the top/bottom images of each axis (fold-0 shared model; see outputs/phase_c/axes.html).
const AXES = [
  { hi: "Crowd-pleaser", lo: "Against the grain" },
  { hi: "Vivid, sensual painting", lo: "Quiet minimalism" },
  { hi: "Raw, gestural marks", lo: "Polished detail" },
  { hi: "Old-master portraits", lo: "The human body" },
  { hi: "Open landscapes", lo: "Intimate figures" },
  { hi: "Baroque drama", lo: "Clean colour fields" },
  { hi: "Busy geometry", lo: "Near-empty canvases" },
  { hi: "Playful, naive figures", lo: "Explosive abstraction" },
  { hi: "Lush florals", lo: "Stark black & white" },
];

let M, U, K, N, REF; // REF = reference crowd for "you vs other people" (data/ref.json, src/app_reference.py)
let S; // session state

// ------------------------------------------------------------------ small linear algebra (n <= 10)
function solve(A, b) { // Gaussian elimination with partial pivoting; A, b are copied
  const n = b.length, a = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    for (let r = c + 1; r < n; r++) {
      const f = a[r][c] / a[c][c];
      for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = a[r][n];
    for (let k = r + 1; k < n; k++) s -= a[r][k] * x[k];
    x[r] = s / a[r][r];
  }
  return x;
}
function inverse(A) {
  const n = A.length;
  const cols = [...Array(n)].map((_, j) => solve(A, [...Array(n)].map((_, i) => +(i === j))));
  return [...Array(n)].map((_, i) => cols.map((c) => c[i]));
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// ------------------------------------------------------------------ model
const row = (i) => U.subarray(i * K, i * K + K);
function score(z, i) { let s = 0; const o = i * K; for (let k = 0; k < K; k++) s += z[k] * U[o + k]; return s; }
function diff(a, b) { const d = new Float64Array(K), oa = a * K, ob = b * K; for (let k = 0; k < K; k++) d[k] = U[oa + k] - U[ob + k]; return d; }

function priorBlocks() {
  const n = K + 1, m = [...M.mu, 0];
  const P = [...Array(n)].map((_, i) => [...Array(n)].map((_, j) => (i < K && j < K ? M.tau * M.Sinv[i][j] : 0)));
  P[K][K] = 1 / (CFG.sideBiasSd * CFG.sideBiasSd);
  return { m, P };
}
const PRIOR = { m: null, P: null };

/** Negative log posterior of theta (up to a constant): logistic loss + Gaussian prior. */
function objective(th, X, answers) {
  const n = K + 1, { m, P } = PRIOR;
  let v = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v += 0.5 * (th[i] - m[i]) * P[i][j] * (th[j] - m[j]);
  answers.forEach((a, r) => {
    let l = 0; for (let i = 0; i < n; i++) l += X[r][i] * th[i];
    v += Math.log1p(Math.exp(-Math.abs(l))) + Math.max(l, 0) - a.y * l; // log(1 + e^l) - y l, overflow-safe
  });
  return v;
}

/** MAP of theta = [z, b] given answers [{top, bottom, y}] (y = 1 if top chosen). Returns {z, b, H}.
 * Newton with backtracking: a full Newton step can overshoot on logistic losses and never recover
 * (seen in real sessions: objective stuck at 687 vs a true minimum of 12.8), so a step is halved until it lowers the objective. */
function fit(answers) {
  const n = K + 1, { m, P } = PRIOR;
  const X = answers.map((a) => [...diff(a.top, a.bottom), 1]);
  let th = m.slice();
  let H, f0 = objective(th, X, answers);
  for (let it = 0; it < 100; it++) {
    const g = new Array(n).fill(0);
    H = P.map((r) => r.slice());
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) g[i] += P[i][j] * (th[j] - m[j]);
    answers.forEach((a, r) => {
      const x = X[r];
      let l = 0; for (let i = 0; i < n; i++) l += x[i] * th[i];
      const p = sigmoid(l), w = p * (1 - p);
      for (let i = 0; i < n; i++) {
        g[i] += (p - a.y) * x[i];
        for (let j = 0; j < n; j++) H[i][j] += w * x[i] * x[j];
      }
    });
    const step = solve(H, g);
    let t = 1, next, f1;
    for (; t > 1e-8; t /= 2) {
      next = th.map((v, i) => v - t * step[i]);
      f1 = objective(next, X, answers);
      if (f1 <= f0) break;
    }
    if (!(f1 <= f0)) break; // no descent possible: already at the minimum (numerically)
    let mx = 0;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(t * step[i]));
    th = next; f0 = f1;
    if (mx < 1e-9) break;
  }
  // H above is the Hessian at the point before the last (tiny) step; fine for the Laplace posterior.
  return { z: th.slice(0, K), b: th[K], H };
}

// ------------------------------------------------------------------ pair choice
function randInt(n) { return Math.floor(Math.random() * n); }
function drawImage(exclude) {
  for (let t = 0; t < 1000; t++) {
    const i = S.pool[randInt(S.pool.length)];
    if (!S.used.has(i) && !exclude.has(i)) return i;
  }
  throw new Error("image pool exhausted");
}
function randomPair(exclude = new Set()) {
  const a = drawImage(exclude);
  const b = drawImage(new Set([...exclude, a]));
  return [a, b];
}
function infoPair(answers, exclude) {
  const f = fit(answers);
  const Hz = inverse(f.H).slice(0, K).map((r) => r.slice(0, K));
  let best = null, bestSc = -1;
  for (let c = 0; c < CFG.nCand; c++) {
    const [a, b] = randomPair(exclude);
    const d = diff(a, b);
    let l = 0; for (let k = 0; k < K; k++) l += f.z[k] * d[k];
    const p = sigmoid(l);
    let v = 0;
    for (let i = 0; i < K; i++) { let s = 0; for (let j = 0; j < K; j++) s += Hz[i][j] * d[j]; v += d[i] * s; }
    const sc = p * (1 - p) * v;
    if (sc > bestSc) { bestSc = sc; best = [a, b]; }
  }
  return best;
}
/** Test pair the model is most sure about: largest |s(a) - s(b)| under the person's z among nCand random unused pairs. */
function surePair() {
  let best = null, bestSc = -1;
  for (let c = 0; c < CFG.nCand; c++) {
    const [a, b] = randomPair();
    const sc = Math.abs(S.sUser[a] - S.sUser[b]);
    if (sc > bestSc) { bestSc = sc; best = [a, b]; }
  }
  return best;
}
/** Contrarian test pair: the personal model is most sure AND the crowd model picks the other image (falls back to surePair). */
function contraPair() {
  let best = null, bestSc = -1;
  for (let c = 0; c < CFG.nCand; c++) {
    const [a, b] = randomPair();
    const d = S.sUser[a] - S.sUser[b];
    if (d * (S.sCrowd[a] - S.sCrowd[b]) >= 0) continue;
    if (Math.abs(d) > bestSc) { bestSc = Math.abs(d); best = [a, b]; }
  }
  return best || surePair();
}
/** Fisher–Yates shuffle in place (sort with a random comparator is biased). */
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
/** Randomise which image goes in slot 0 (top / left). */
function place([a, b]) { return Math.random() < 0.5 ? { top: a, bottom: b } : { top: b, bottom: a }; }

const preloaded = new Map();
function preload(i) {
  if (preloaded.has(i)) return preloaded.get(i);
  const im = new Image();
  im.decoding = "async";
  im.src = imgUrl(i);
  const p = im.decode().catch(() => {});
  preloaded.set(i, p);
  if (preloaded.size > 60) preloaded.delete(preloaded.keys().next().value);
  return p;
}

// ------------------------------------------------------------------ rendering
function showScreen(id) {
  for (const s of document.querySelectorAll(".screen")) s.hidden = s.id !== id;
}
const slots = [$("slot0"), $("slot1")];

async function showPair(pr) {
  const pair = $("pair");
  pair.classList.remove("shown");
  for (const s of slots) {
    s.classList.remove("picked", "model");
    s.querySelector(".tag").className = "tag";
  }
  $("verdict").textContent = "";
  $("verdict").className = "verdict";
  $("next").hidden = true;
  const ids = [pr.top, pr.bottom];
  await Promise.all(ids.map(preload));
  ids.forEach((i, k) => { slots[k].querySelector("img").src = imgUrl(i); });
  await Promise.all(slots.map((s) => s.querySelector("img").decode().catch(() => {})));
  // both images appear together: neither gets a head start in looking time
  requestAnimationFrame(() => {
    pair.classList.add("shown");
    S.shownAt = performance.now();
    S.locked = false;
  });
}

function setHeader() {
  const train = S.phase === "train";
  const n = train ? S.answers.length : S.tests.length;
  const total = train ? CFG.nTrain : CFG.nTest;
  $("phase").textContent = train ? "Learning your taste" : "Model guesses first";
  const hits = S.tests.filter((t) => t.choice !== undefined && t.choice === t.pred).length;
  const done = S.tests.filter((t) => t.choice !== undefined).length;
  $("count").textContent = train ? `${n + 1} / ${total}` : `${Math.min(n, total)} / ${total}` + (done ? ` · ✓ ${hits}` : "");
  $("progress-fill").style.width = `${(100 * (train ? n : done)) / total}%`;
}

// ------------------------------------------------------------------ training phase
function nextTrainPair() {
  const n = S.answers.length;
  let pr = S.ahead ? (S.ahead.any ?? S.ahead[S.answers[n - 1].y]) : null;
  if (!pr) pr = place(isRandomQ(n) ? randomPair() : infoPair(S.answers, new Set()));
  S.ahead = null;
  return pr;
}
/** While the person looks at the current pair, compute the next pair for both possible answers and preload it. */
function planAhead(cur) {
  const n = S.answers.length + 1; // answers after this one
  if (n >= CFG.nTrain) return;
  const ex = new Set([cur.top, cur.bottom]);
  if (isRandomQ(n)) {
    const pr = place(randomPair(ex));
    S.ahead = { any: pr };
    preload(pr.top); preload(pr.bottom);
    return;
  }
  S.ahead = {};
  for (const y of [1, 0]) {
    const hyp = [...S.answers, { top: cur.top, bottom: cur.bottom, y }];
    const pr = place(infoPair(hyp, ex));
    S.ahead[y] = pr;
    preload(pr.top); preload(pr.bottom);
  }
}

async function trainStep() {
  setHeader();
  const pr = nextTrainPair();
  S.cur = pr;
  S.used.add(pr.top); S.used.add(pr.bottom);
  await showPair(pr);
  setTimeout(() => { if (S.cur === pr) planAhead(pr); }, 30);
}

function onPick(k) {
  if (!S || S.locked || !S.cur) return;
  const dt = performance.now() - S.shownAt;
  if (dt < CFG.minViewMs) return;
  S.locked = true;
  slots[k].classList.add("picked");
  if (S.phase === "train") {
    S.answers.push({ top: S.cur.top, bottom: S.cur.bottom, y: k === 0 ? 1 : 0, ms: Math.round(dt) });
    if (!S.ahead && S.answers.length < CFG.nTrain) planAhead(S.cur); // tapped before planning finished
    if (S.answers.length % 10 === 0) savePartial();
    setTimeout(() => {
      if (S.answers.length >= CFG.nTrain) finishTraining();
      else trainStep();
    }, 160);
  } else {
    revealTest(k, dt);
    if (S.tests.length % 5 === 0) savePartial();
  }
}

// ------------------------------------------------------------------ test phase
function finishTraining() {
  const f = fit(S.answers);
  S.z = f.z; S.b = f.b;
  S.sUser = new Float64Array(N); S.sCrowd = new Float64Array(N);
  for (let i = 0; i < N; i++) { S.sUser[i] = score(S.z, i); S.sCrowd[i] = score(M.mu, i); }
  S.phase = "test";
  S.tests = [];
  // which test positions get a "sure" / "contra" pair (random positions, so they are not all at the start or end)
  const order = shuffle([...Array(CFG.nTest).keys()]);
  S.kindAt = new Map(order.slice(0, CFG.nSure + CFG.nContra).map((p, j) => [p, j < CFG.nSure ? "sure" : "contra"]));
  S.nextTest = makeTest();
  showScreen("ready");
}
function makeTest() {
  const kind = S.kindAt.get(S.tests.length) || "random"; // index of the test this pair will become
  const pr = place(kind === "sure" ? surePair() : kind === "contra" ? contraPair() : randomPair());
  const st = score(S.z, pr.top), sb = score(S.z, pr.bottom);
  const ct = score(M.mu, pr.top), cb = score(M.mu, pr.bottom);
  preload(pr.top); preload(pr.bottom);
  // the guess is fixed here, before the pair is on screen
  return { ...pr, kind, pred: st >= sb ? 0 : 1, conf: sigmoid(Math.abs(st - sb)), cons: ct >= cb ? 0 : 1 };
}
async function testStep() {
  const t = S.nextTest;
  S.tests.push(t);
  S.cur = t;
  S.used.add(t.top); S.used.add(t.bottom);
  setHeader();
  await showPair(t);
  if (S.tests.length < CFG.nTest) S.nextTest = makeTest();
}
function revealTest(k, dt) {
  const t = S.cur;
  t.choice = k; t.ms = Math.round(dt);
  slots[t.pred].classList.add("model");
  [0, 1].forEach((j) => {
    const tag = slots[j].querySelector(".tag");
    const parts = [];
    if (j === k) parts.push("Your pick");
    if (j === t.pred) parts.push("Model's guess");
    if (parts.length) { tag.textContent = parts.join(" · "); tag.className = "tag on" + (j === t.pred ? " model" : ""); }
  });
  const hit = k === t.pred;
  $("verdict").textContent = hit ? "✓ Called it" : "✗ Missed";
  $("verdict").className = "verdict " + (hit ? "good" : "bad");
  setHeader();
  const last = S.tests.length >= CFG.nTest;
  $("next").textContent = last ? "See results" : "Next";
  $("next").hidden = false;
}

// ------------------------------------------------------------------ results
function traits(z) {
  return z.map((v, k) => ({ k, dev: (v - M.mu[k]) / Math.sqrt(M.Sig[k][k]) }))
    .sort((a, b) => Math.abs(b.dev) - Math.abs(a.dev))
    .map((t) => ({ ...t, name: t.dev > 0 ? AXES[t.k].hi : AXES[t.k].lo, vs: t.dev > 0 ? AXES[t.k].lo : AXES[t.k].hi }));
}
const artist = (i) => M.path[i].split("/").pop().split("_")[0];
/** Keep the first n of an ordered list, at most one image per artist. */
function oneEach(ordered, n) {
  const seen = new Set(), out = [];
  for (const i of ordered) {
    if (seen.has(artist(i))) continue;
    seen.add(artist(i)); out.push(i);
    if (out.length === n) break;
  }
  return out;
}
function topBy(fn, n, filter) {
  const idx = [];
  for (const i of S.pool) if (filter(i)) idx.push(i);
  idx.sort((a, b) => fn(b) - fn(a));
  return oneEach(idx, n);
}
function axisExamples(t, n) {
  const sign = t.dev > 0 ? 1 : -1;
  const idx = S.pool.slice().sort((a, b) => sign * (U[b * K + t.k] - U[a * K + t.k]));
  return oneEach(idx.slice(0, 300).sort((a, b) => S.sUser[b] - S.sUser[a]), n);
}
function tasteMatch(z1, z2) { // correlation of the personal part of the scores over a fixed image sample
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, n = 0;
  for (let i = 0; i < N; i += 3) {
    let x = 0, y = 0; const o = i * K;
    for (let k = 0; k < K; k++) { x += (z1[k] - M.mu[k]) * U[o + k]; y += (z2[k] - M.mu[k]) * U[o + k]; }
    sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; n++;
  }
  const r = (sxy - (sx * sy) / n) / Math.sqrt((sxx - (sx * sx) / n) * (syy - (sy * sy) / n));
  return Math.round(50 * (1 + r));
}
/** Share of the reference crowd (data/ref.json: LAPIS raters fitted from 50 app-style picks) below x, in 0..100. */
function pctBelow(sorted, x) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] < x) lo = m + 1; else hi = m; }
  return Math.min(99, Math.max(1, Math.round((100 * lo) / sorted.length))); // 1..99: the reference crowd is a sample
}
/** How much more of trait t this person has than other people (percent), in the trait's own direction. */
const traitPct = (t) => (t.dev > 0 ? pctBelow(REF.axes[t.k], t.dev) : 100 - pctBelow(REF.axes[t.k], t.dev));
/** One line comparing the whole taste with other people, framed by which side of the median it falls. */
const uniqLine = (p) => (p >= 50 ? `more unusual than ${p}% of people` : `more mainstream than ${100 - p}% of people`);
/** How unusual the whole taste is: share of people with a taste closer to the average (Mahalanobis distance of z). */
function uniqPct(z) {
  let d2 = 0;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) d2 += (z[i] - M.mu[i]) * M.Sinv[i][j] * (z[j] - M.mu[j]);
  return pctBelow(REF.uniq, Math.sqrt(d2));
}
function img(i) { const im = document.createElement("img"); im.src = imgUrl(i); im.alt = ""; im.loading = "lazy"; return im; }

function showResults() {
  const hits = S.tests.filter((t) => t.choice === t.pred).length;
  const crowd = S.tests.filter((t) => t.choice === t.cons).length;
  S.hits = hits;
  track("Finished", { score: hits, crowd, challenge: !!S.challenge });
  $("r-score").textContent = hits;
  $("r-sub").textContent = `It learned your taste from ${CFG.nTrain} picks, then guessed ${hits} of your next ${CFG.nTest} before you chose.`;
  const T = traits(S.z);
  S.traits = T;
  $("r-label").textContent = `${T[0].name} × ${T[1].name}`;
  S.uniq = uniqPct(S.z);
  $("r-unique").textContent = `Your taste is ${uniqLine(S.uniq)}.`;
  const box = $("r-axes"); box.replaceChildren();
  for (const t of T.slice(0, 3)) {
    const d = document.createElement("div"); d.className = "axis";
    d.innerHTML = `<div class="axis-name"><span></span><span></span></div><div class="strip"></div>`;
    d.querySelector(".axis-name span").textContent = t.name;
    const pct = traitPct(t);
    d.querySelector(".axis-name span:last-child").textContent = pct >= 50 ? `more than ${pct}% of people` : "";
    for (const i of axisExamples(t, 4)) d.querySelector(".strip").append(img(i));
    box.append(d);
  }
  S.picks = topBy((i) => S.sUser[i], 6, (i) => !S.used.has(i));
  $("r-picks").replaceChildren(...S.picks.map((i) => {
    const b = document.createElement("button");
    b.className = "pick"; b.setAttribute("aria-label", "Open this artwork");
    b.append(img(i));
    b.addEventListener("click", () => openArt(i));
    return b;
  }));
  if (S.challenge) {
    const m = tasteMatch(S.z, S.challenge.z);
    const who = S.challenge.n || "your friend";
    $("r-match").innerHTML = `You and <span></span>: <b>${m}%</b> taste match<br><span class="hint"></span>`;
    $("r-match").querySelector("span").textContent = who;
    $("r-match").querySelector(".hint").textContent = `Their model guessed ${S.challenge.s}/${S.challenge.t}, yours ${hits}/${CFG.nTest}.`;
    $("r-match").hidden = false;
  }
  try { $("r-name").value = localStorage.getItem("taste_name") || ""; } catch {}
  showScreen("results");
  $("results").scrollTop = 0;
  saveSession();
}

// ------------------------------------------------------------------ artwork viewer (suggested picks)
/** Artist, title and year from the LAPIS (WikiArt-style) filename, e.g. "abstract/aaron-siskind_new-york-2-1948.jpg". */
function credit(i) {
  const stem = M.path[i].split("/").pop().replace(/\.[a-z]+$/i, "");
  const [a, ...rest] = stem.split("_");
  const cap = (t) => t.split("-").filter(Boolean).map((w) => w.replace(/(^|\.)([a-z])/g, (_, p, ch) => p + ch.toUpperCase())).join(" ");
  let t = rest.join("_");
  const m = t.match(/-(\d{4})$/);
  const year = m ? m[1] : "";
  if (m) t = t.slice(0, -5);
  return { artist: cap(a), title: cap(t) || "Untitled", year, file: `${stem}.jpg` };
}
function openArt(i) {
  const c = credit(i);
  S.viewing = i;
  $("art-img").src = imgUrl(i);
  $("art-title").textContent = c.year ? `${c.title}, ${c.year}` : c.title;
  $("art-artist").textContent = c.artist;
  $("art-open").href = imgUrl(i);
  $("art").hidden = false;
  track("Open artwork");
}
async function saveArt() {
  const i = S.viewing, c = credit(i);
  track("Download artwork");
  try {
    const blob = await fetch(imgUrl(i)).then((r) => { if (!r.ok) throw new Error(r.status); return r.blob(); });
    const file = new File([blob], c.file, { type: blob.type || "image/jpeg" });
    if (navigator.canShare && navigator.canShare({ files: [file] })) { // iPhone: share sheet with "Save Image"
      try { await navigator.share({ files: [file] }); return; } catch (e) { if (e.name === "AbortError") return; }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = c.file;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  } catch {
    window.open(imgUrl(i), "_blank"); // last resort: open it; long-press to save
  }
}

// ------------------------------------------------------------------ share
function b64url(s) { return btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""); }
function unb64url(s) { return decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/")))); }
function shareLink(name) {
  const z = S.z.map((v, k) => Math.round((1000 * (v - M.mu[k])) / Math.sqrt(M.Sig[k][k])));
  const payload = b64url(JSON.stringify({ n: name, s: S.hits, t: CFG.nTest, z }));
  return `${location.origin}${location.pathname}#c=${payload}`;
}
function readChallenge() {
  const m = location.hash.match(/#c=([\w-]+)/);
  if (!m) return null;
  try {
    const o = JSON.parse(unb64url(m[1]));
    if (!Array.isArray(o.z) || o.z.length !== K) return null;
    const t = Math.max(1, Math.min(100, o.t | 0 || 10)); // older links have no t: they were out of 10
    return { n: String(o.n || "").slice(0, 24), t, s: Math.max(0, Math.min(t, o.s | 0)),
             z: o.z.map((v, k) => M.mu[k] + (v / 1000) * Math.sqrt(M.Sig[k][k])) };
  } catch { return null; }
}
function loadForCanvas(i) {
  return new Promise((res) => {
    const im = new Image();
    if (/^https?:/.test(CFG.imgBase)) im.crossOrigin = "anonymous"; // Blob store on another origin
    im.onload = () => res(im); im.onerror = () => res(null);
    im.src = imgUrl(i);
  });
}
function wrap(ctx, text, x, y, maxW, lh) {
  const words = text.split(" "); let line = "";
  for (const w of words) {
    const t = line ? line + " " + w : w;
    if (ctx.measureText(t).width > maxW && line) { ctx.fillText(line, x, y); line = w; y += lh; } else line = t;
  }
  ctx.fillText(line, x, y);
  return y + lh;
}
async function makeCard() {
  const W = 1080, H = 1350, c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  const font = '-apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif';
  g.fillStyle = "#161616"; g.fillRect(0, 0, W, H);
  g.fillStyle = "#9a9a96"; g.font = `600 34px ${font}`;
  g.fillText(`AN AI LEARNED MY TASTE IN ART FROM ${CFG.nTrain} PICKS`, 80, 130);
  g.fillStyle = "#ededeb"; g.font = `700 260px ${font}`;
  g.fillText(String(S.hits), 70, 390);
  const wd = g.measureText(String(S.hits)).width;
  g.fillStyle = "#6b6b68"; g.font = `700 120px ${font}`;
  g.fillText(`/${CFG.nTest}`, 80 + wd, 390);
  g.fillStyle = "#ededeb"; g.font = `500 46px ${font}`;
  let y = wrap(g, `then guessed ${S.hits} of my next ${CFG.nTest} picks before I made them.`, 80, 470, W - 160, 58);
  g.fillStyle = "#9a9a96"; g.font = `600 30px ${font}`;
  g.fillText("MY TASTE", 80, y + 40);
  g.fillStyle = "#ededeb"; g.font = `700 54px ${font}`;
  y = wrap(g, `${S.traits[0].name} × ${S.traits[1].name}`, 80, y + 108, W - 160, 64);
  g.fillStyle = "#9a9a96"; g.font = `500 36px ${font}`;
  const u = uniqLine(S.uniq);
  y = wrap(g, u[0].toUpperCase() + u.slice(1), 80, y + 4, W - 160, 46);
  // three artworks the model picked for me
  const picks = topBy((i) => S.sUser[i], 3, (i) => !S.used.has(i));
  const ims = await Promise.all(picks.map(loadForCanvas));
  const gap = 24, tw = (W - 160 - 2 * gap) / 3, top = y + 30, th = Math.min(tw * 1.3, H - 200 - top);
  ims.forEach((im, j) => {
    const x = 80 + j * (tw + gap);
    g.fillStyle = "#222221"; g.fillRect(x, top, tw, th);
    if (!im) return;
    const s = Math.max(tw / im.width, th / im.height), sw = tw / s, sh = th / s;
    g.drawImage(im, (im.width - sw) / 2, (im.height - sh) / 2, sw, sh, x, top, tw, th);
  });
  g.fillStyle = "#9a9a96"; g.font = `500 30px ${font}`;
  g.fillText("Picked for me by the model", 80, top + th + 44);
  g.fillStyle = "#ededeb"; g.font = `700 40px ${font}`;
  g.fillText(`Can it read you? ${location.host}`, 80, H - 60);
  return new Promise((res) => c.toBlob(res, "image/png"));
}
async function share() {
  const name = $("r-name").value.trim();
  try { localStorage.setItem("taste_name", name); } catch {}
  const btn = $("share"); btn.disabled = true; btn.textContent = "Making your card…";
  const link = shareLink(name);
  const blob = await makeCard();
  btn.disabled = false; btn.textContent = "Share my result";
  const file = new File([blob], "my-taste.png", { type: "image/png" });
  const text = `An AI learned my taste in art from ${CFG.nTrain} picks and guessed ${S.hits}/${CFG.nTest} of my next ones. ` +
    `My taste: ${S.traits[0].name} × ${S.traits[1].name}, ${uniqLine(S.uniq)}. Can it read you?`;
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file], text: `${text} ${link}` }); return; } catch (e) { if (e.name === "AbortError") return; }
  }
  // fallback (e.g. plain http on the LAN, where the Web Share API is unavailable)
  $("sheet-img").src = URL.createObjectURL(blob);
  $("sheet-link").value = link;
  try { await navigator.clipboard.writeText(`${text} ${link}`); } catch {}
  $("sheet").hidden = false;
}

// ------------------------------------------------------------------ session log (kept on the device for now)
function buildRec(partial) {
  const done = S.tests.filter((t) => t.choice !== undefined);
  return {
    v: 2, id: S.id, partial, at: new Date().toISOString(), model: M.source, fromChallenge: !!S.challenge,
    tag: CFG.tag, nTrain: CFG.nTrain, nTest: CFG.nTest, nSure: CFG.nSure, nContra: CFG.nContra, nRandom: CFG.nRandom, mixRandom: CFG.mixRandom,
    // q = question index; rnd = whether it was a random (not infogain) pair; paths join to data/lapis/images/
    train: S.answers.map((a, q) => ({ top: M.image_id[a.top], bottom: M.image_id[a.bottom], topPath: M.path[a.top],
                                       bottomPath: M.path[a.bottom], y: a.y, ms: a.ms, rnd: isRandomQ(q) })),
    test: done.map((t) => ({ top: M.image_id[t.top], bottom: M.image_id[t.bottom], topPath: M.path[t.top],
                                 bottomPath: M.path[t.bottom], choice: t.choice, pred: t.pred,
                                 conf: +t.conf.toFixed(3), cons: t.cons, ms: t.ms, kind: t.kind,
                                 sure: t.kind === "sure" })),
    z: S.z ? S.z.map((v) => +v.toFixed(5)) : null, sideBias: S.z ? +S.b.toFixed(4) : null, hits: S.hits ?? null,
    topRate: +((S.answers.filter((a) => a.y === 1).length + done.filter((t) => t.choice === 0).length) /
               Math.max(1, S.answers.length + done.length)).toFixed(3),
  };
}
/** Progress checkpoint (same id, overwritten on the server) so a long session survives a closed tab. */
function savePartial() {
  fetch("api/session", { method: "POST", headers: { "Content-Type": "application/json" },
                         body: JSON.stringify(buildRec(true)) }).catch(() => {});
}
function saveSession() {
  const rec = buildRec(false);
  try {
    const all = JSON.parse(localStorage.getItem("taste_sessions") || "[]");
    all.push(rec);
    localStorage.setItem("taste_sessions", JSON.stringify(all));
  } catch {}
  // Local server (serve.py) writes the record to outputs/user_taste/; on Vercel, api/session.js stores it in a private Blob store.
  // The record is anonymous: random id, picks, timings and the fitted z; no name, IP or device info.
  fetch("api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(rec) })
    .then((r) => { $("r-saved").textContent = r.ok ? "Your picks were saved anonymously for research." : "Server did not save this session."; })
    .catch(() => { $("r-saved").textContent = "Server not reachable: session kept on this device only."; });
}

// ------------------------------------------------------------------ boot
function newSession() {
  const pool = [...Array(N).keys()];
  S = { id: new Date().toISOString().replace(/[:.]/g, "-") + "_" + Math.random().toString(36).slice(2, 8), phase: "train", answers: [], tests: [], used: new Set(), pool, ahead: null, locked: true,
        challenge: readChallenge() };
}
async function boot() {
  const [meta, buf, ref] = await Promise.all([
    fetch("data/meta.json").then((r) => r.json()),
    fetch("data/U.bin").then((r) => r.arrayBuffer()),
    fetch("data/ref.json").then((r) => r.json()),
  ]);
  REF = ref;
  M = meta; U = new Float32Array(buf); K = M.k; N = M.n;
  Object.assign(PRIOR, priorBlocks());
  document.querySelectorAll(".n-train").forEach((e) => { e.textContent = CFG.nTrain; });
  document.querySelectorAll(".n-test").forEach((e) => { e.textContent = CFG.nTest; });
  newSession();
  if (S.challenge) {
    $("challenge").textContent = `${S.challenge.n || "A friend"}'s taste model guessed ${S.challenge.s}/${S.challenge.t}. Can it read you?`;
    $("challenge").hidden = false;
  }
  const first = place(randomPair());
  S.ahead = { any: first };
  preload(first.top); preload(first.bottom);
  $("start").disabled = false; $("start").textContent = "Start";
}

slots.forEach((s, k) => s.addEventListener("click", () => onPick(k)));
$("start").addEventListener("click", () => { track("Start", { challenge: !!S.challenge }); showScreen("play"); trainStep(); });
$("go-test").addEventListener("click", () => { track("Test start"); showScreen("play"); testStep(); });
$("next").addEventListener("click", () => { if (S.tests.length >= CFG.nTest) showResults(); else testStep(); });
$("share").addEventListener("click", () => { track("Share"); share(); });
$("sheet-close").addEventListener("click", () => { $("sheet").hidden = true; });
$("art-save").addEventListener("click", saveArt);
$("art-close").addEventListener("click", () => { $("art").hidden = true; });
$("art").addEventListener("click", (e) => { if (e.target === $("art")) $("art").hidden = true; });
$("restart").addEventListener("click", () => {
  track("Play again");
  newSession();
  const first = place(randomPair());
  S.ahead = { any: first };
  showScreen("play"); trainStep();
});
boot().catch((e) => { $("start").textContent = "Could not load the model"; console.error(e); });
