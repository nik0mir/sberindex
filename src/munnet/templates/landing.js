// Лендинг «Корзина и регион» (пакет munnet): экран 0 (ответ и карта ячеек), поиск муниципалитета и карточка МО.
// Страница ничего не считает, кроме форматирования, поиска по названию и перевода указателя в ячейку.
// Данные: встроенные <script type="application/json"> (story, names, meta, data-*) или data/*.json (fetch).

const doc = document;
doc.documentElement.classList.add("js");
const $ = (s, r = doc) => r.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const nf = new Intl.NumberFormat("ru-RU");

function readJSON(id) {
  const el = doc.getElementById(id);
  if (!el || !el.textContent.trim()) return null;
  try { return JSON.parse(el.textContent); } catch (e) { console.warn("munnet: не разобран", id); return null; }
}
async function loadData(name) {
  const inl = readJSON("data-" + name);
  if (inl) return inl;
  if (location.protocol === "file:") return null; // по file:// fetch не работает — нужны встроенные блоки
  try {
    const r = await fetch(`data/${name}.json`);
    return r.ok ? await r.json() : null;
  } catch (e) { return null; }
}
// Колонки {id: [...], n: [...]} или массив записей -> массив записей.
function rowsOf(x) {
  if (!x) return [];
  if (Array.isArray(x)) return x;
  const keys = Object.keys(x);
  const n = keys.length ? x[keys[0]].length : 0;
  const out = new Array(n);
  for (let i = 0; i < n; i++) { const o = {}; for (const k of keys) o[k] = x[k][i]; out[i] = o; }
  return out;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const cap1 = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
// ru-text: неразрывный пробел после однобуквенных слов и перед тире (как landing.nbsp; только показ)
const NB = "\u00a0";
const nbsp = (s) => String(s ?? "").replace(/(^|[\s>(«])([вксоуиаяВКСОУИАЯ]) /g, "$1$2" + NB).replace(/ — /g, NB + "— ");
function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b > 1 && b < 5) return few;
  if (b === 1) return one;
  return many;
}

const story = readJSON("story") || {};
const meta = readJSON("meta") || story.meta || {};
const view = story.view || {};
const names = story.names || {};
const S0 = story.screen0 || {};
const CH = story.chapters || {};
const CARD = story.card || {}; // необязательные строки карточки от landing.py (зависят от вердиктов)
const T3 = (story.verdicts || {}).T3_reliable_placebo;
const SHAPES = view.shapes || { 1: "●", 2: "▲", 3: "■", 4: "◆" };
const PART_KEYS = ["food", "marketplace", "transport", "health", "cafe", "other"];
const PART_LABELS = story.part_labels || {
  food: "продукты", marketplace: "маркетплейсы", transport: "транспорт",
  health: "здоровье", cafe: "кафе и рестораны", other: "прочее",
};
const MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
const SRC = "Источник: СберИндекс (CC BY-SA 4.0); Росстат, БД ПМО в обработке «Если быть точным» (CC BY 4.0); ФНС, 5-НДФЛ в обработке «Если быть точным» (лицензия на странице набора не указана); расчёт «Корзина и регион».";

// палитра типов — из story.view (порядковая или номинальная по вердикту T1)
for (const [t, c] of Object.entries(view.type_colors || {})) doc.documentElement.style.setProperty("--t" + t, c);

const typeName = (t) => (t == null ? "нет типа" : (names.final || {})[t] || `Тип ${t}`);
function fig(t) {
  if (t == null) return '<i class="fig fig-t0" aria-hidden="true"></i>';
  return `<i class="fig fig-t${t}" aria-hidden="true">${SHAPES[t] || ""}</i>`;
}
const typeHTML = (t) => `${fig(t)}<span>${esc(typeName(t))}</span>`;

// --- плашка режима и подвал -------------------------------------------------------------------------------
{
  const b = $("#banner"); // текст плашки вписан при сборке; здесь только её высота для липкой шапки
  if (b && !b.hidden) doc.documentElement.style.setProperty("--banner-h", b.offsetHeight + "px");
}

// --- шапка на телефоне: разделы — по кнопке «Меню» (без JS ссылки прокручиваются) -----------------------------
{
  const btn = $("#nav-toggle"), list = $("#nav-list");
  if (btn && list) {
    btn.hidden = false;
    const set = (on) => { list.classList.toggle("open", on); btn.setAttribute("aria-expanded", String(on)); };
    btn.addEventListener("click", () => set(!list.classList.contains("open")));
    list.addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
    doc.addEventListener("click", (e) => { if (!e.target.closest(".masthead nav")) set(false); });
    doc.addEventListener("keydown", (e) => { if (e.key === "Escape" && list.classList.contains("open")) { set(false); btn.focus(); } });
  }
}

// --- экран 0: тексты вписаны при сборке (landing.screen0_html); на телефоне пункты 2–3 — по кнопке --------------
{
  const ol = $("#answer-points");
  const extra = ol ? ol.querySelectorAll("li.extra").length : 0;
  if (ol && extra) {
    const tl = doc.createElement("li");
    tl.className = "points-more";
    const label = `Ещё ${extra === 2 ? "два вывода" : "выводы"}`;
    tl.innerHTML = `<button type="button" class="tool tool-text" aria-expanded="false">${label}</button>`;
    tl.querySelector("button").addEventListener("click", (e) => {
      const on = ol.classList.toggle("show-extra");
      e.currentTarget.setAttribute("aria-expanded", String(on));
      e.currentTarget.textContent = on ? "Свернуть" : label;
    });
    ol.children[0].after(tl);
  }
}

// --- поиск: ARIA combobox (WAI-ARIA APG, список с aria-activedescendant) ---------------------------------------
const STOP = [/муниципальн(ый|ого) (район|округ)/g, /городск(ой|ого) округ/g, /внутригородская территория/g,
  /города федерального значения/g, /городское поселение/g, /сельское поселение/g];
function norm(s) {
  let x = String(s || "").toLowerCase().replace(/ё/g, "е").replace(/[«»"()]/g, " ");
  for (const re of STOP) x = x.replace(re, " ");
  return x.replace(/[-‐–—]/g, " ").replace(/\s+/g, " ").trim();
}
// Индекс поиска: колонки id, n, r, t; регион — номер в списке rl (или строка в старом формате).
const NAMES_RAW = readJSON("names") || {};
const REGION_LIST = NAMES_RAW.rl || null;
delete NAMES_RAW.rl;
const NAMES = rowsOf(NAMES_RAW).map((r0) => {
  const r = REGION_LIST && typeof r0.r === "number" ? { ...r0, r: REGION_LIST[r0.r] } : r0;
  const nn = norm(r.n), nr = norm(r.r), nc = norm(r.c); // c — центр МО, если landing.py его положил
  return { ...r, nn, nc, words: (nn + " " + nc).trim().split(" "), all: (nn + " " + nc + " " + nr).trim().split(" ") };
});
// Поиск: по началу названия, по началу слова, по подстроке; несколько слов — каждое с начала слова
// в названии или регионе («октябрьский костром»), так различаются одноимённые МО разных регионов.
function search(q) {
  const nq = norm(q);
  if (!nq) return [];
  const found = searchExact(nq);
  if (found.length || nq.length < 5) return found;
  // «Кириши» -> «Киришский район»: без окончания (одна-две последние буквы), не короче 4 букв
  for (const cut of [1, 2]) {
    const r = searchExact(nq.slice(0, -cut));
    if (r.length && nq.length - cut >= 4) return r;
  }
  return [];
}
function searchExact(nq) {
  const toks = nq.split(" ");
  const out = [];
  for (const r of NAMES) {
    let s = -1;
    if (r.nn.startsWith(nq) || (r.nc && r.nc.startsWith(nq))) s = 0;
    else if (r.words.some((w) => w.startsWith(nq))) s = 1;
    else if (r.nn.includes(nq)) s = 2;
    else if (toks.length > 1 && toks.every((t) => r.all.some((w) => w.startsWith(t)))) s = 3;
    if (s >= 0) out.push([s, r.nn.length, r]);
  }
  out.sort((a, b) => a[0] - b[0] || a[1] - b[1] || String(a[2].r).localeCompare(String(b[2].r), "ru"));
  return out.map((x) => x[2]);
}
const MAX_OPTS = 12;
let openSearch = () => {}; // поиск с готовым запросом (адрес ?mo=<название> с несколькими совпадениями)
{
  const input = $("#search-input"), list = $("#search-list"), status = $("#search-status");
  let items = [], active = -1, chosen = null; // порция 6m: выбранное не открывает список заново
  const optLabel = (r) => `${r.n} — ${r.r} — ${typeName(r.t)}`;
  function close() { list.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; }
  function setActive(i) {
    active = i;
    [...list.children].forEach((li, j) => li.setAttribute("aria-selected", String(j === i)));
    if (i >= 0 && list.children[i]) {
      input.setAttribute("aria-activedescendant", list.children[i].id);
      list.children[i].scrollIntoView({ block: "nearest" });
    } else input.removeAttribute("aria-activedescendant");
  }
  function render() {
    const all = search(input.value);
    items = all.slice(0, MAX_OPTS);
    const q = norm(input.value);
    if (!q) { close(); status.textContent = ""; return; }
    list.innerHTML = items.length
      ? nbsp(items.map((r, i) => `<li role="option" id="opt-${i}" data-id="${r.id}" aria-selected="false" aria-label="${esc(optLabel(r))}">${fig(r.t)}<span class="opt-name">${esc(r.n)}</span><span class="opt-meta">${esc(r.r)} — ${esc(typeName(r.t))}</span></li>`).join(""))
      : '<li class="empty" role="option" aria-disabled="true">Ничего не нашлось. Попробуйте начало названия без «район» или «округ»' + (S0.search_none ? `<small>${esc(S0.search_none)}.</small>` : "") + "</li>";
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    const n = all.length;
    if (n > MAX_OPTS) list.insertAdjacentHTML("beforeend", `<li class="empty" role="option" aria-disabled="true">Ещё ${nf.format(n - MAX_OPTS)}: добавьте к названию регион, например «${esc(input.value.trim())} ${esc(String(items[items.length - 1].r).split(" ")[0])}»</li>`);
    status.textContent = n ? `Найдено ${nf.format(n)} ${plural(n, "вариант", "варианта", "вариантов")}${n > MAX_OPTS ? `, показаны первые ${MAX_OPTS}` : ""}` : "Ничего не нашлось";
    setActive(-1);
  }
  function choose(i) {
    const r = items[i];
    if (!r) return;
    input.value = r.n;
    chosen = r.n;
    close();
    go(r.id);
  }
  openSearch = (q) => {
    input.value = q;
    input.focus({ preventScroll: true });
    input.scrollIntoView({ block: "center" });
    render();
  };
  input.addEventListener("input", render);
  input.addEventListener("focus", () => { if (input.value && input.value !== chosen) render(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); if (list.hidden) render(); setActive(Math.min(active + 1, items.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(Math.max(active - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); choose(active >= 0 ? active : 0); }
    else if (e.key === "Escape") { if (!list.hidden) { e.stopPropagation(); close(); } else input.value = ""; }
    else if (e.key === "Home" && !list.hidden && active >= 0) { e.preventDefault(); setActive(0); }
    else if (e.key === "End" && !list.hidden && active >= 0) { e.preventDefault(); setActive(items.length - 1); }
  });
  list.addEventListener("mousedown", (e) => e.preventDefault()); // фокус остаётся в поле
  list.addEventListener("click", (e) => {
    const li = e.target.closest("li[data-id]");
    if (li) choose([...list.children].indexOf(li));
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
}

// --- данные карточки и карты ---------------------------------------------------------------------------------
const MO = new Map();
const TYPES = new Map();
const CELLS = new Map(); // "q,r" -> запись узла (территориальный, город или без типа)
let HEX = null;
let CITY_N = new Map();

// --- адрес #mo=<id>: открыть, вернуться, закрыть -------------------------------------------------------------
let current = null;
let trail = []; // откуда пришли по ссылкам «сопоставимых»
let opener = null;
function parseHash() {
  const m = /(?:^|[#&])mo=(\d+)/.exec(location.hash);
  return m ? Number(m[1]) : null;
}
function go(id, fromCard = false) {
  if (!fromCard) trail = [];
  else if (current != null) trail.push(current);
  if (!card.contains(doc.activeElement)) opener = doc.activeElement;
  if (parseHash() === id) route(); else location.hash = "mo=" + id;
}
window.addEventListener("hashchange", () => {
  const id = parseHash();
  if (id != null && trail.length && trail[trail.length - 1] === id) trail.pop();
  route();
});

const card = $("#card");
const cardBody = $("#card-body");
// объёмная карта первого экрана (landing3d.js) слушает выбор и сама открывает карточку через window.munnet.go
// simb — соседи по своему региону (набор B T7), sim — похожие по тратам в других регионах (набор продукта T7):
// объёмная карта рисует к соседям главные дуги, к похожим — второстепенные (порция 5b)
// порция 6e: net — соседи по основной сети корзин (mo.json, поле nb); при включённом переключателе «Соседи по сети
// корзин» карта рисует дуги только к ним (net: true), иначе — к соседям по региону и похожим по тратам
function announce(id, sim = [], simb = [], net = null) { doc.dispatchEvent(new CustomEvent("munnet:select", { detail: { id, sim, simb, net } })); }
let netOn = false; // переключатель карточки; сбрасывается при выборе другого муниципалитета
window.munnet = { go: (id) => go(id), ready: () => MO.size > 0, openParam: () => openParam() };
// адрес ?mo=<номер или название> (порция 6c): номер — сразу карточка; название — как в поиске (начало названия
// без «район», «округ»; «Кириши» -> «Киришский»): одно совпадение (или одно точное) — карточка, несколько —
// открывается поиск с этим запросом, а не случайный муниципалитет. Срабатывает один раз, после загрузки данных.
let paramDone = false;
function openParam() {
  if (paramDone || !MO.size) return null;
  paramDone = true;
  const v = (new URLSearchParams(location.search).get("mo") || "").trim();
  if (!v || parseHash() != null) return null;
  if (/^\d+$/.test(v) && MO.has(Number(v))) { go(Number(v)); return Number(v); }
  const found = search(v), nq = norm(v);
  const exact = found.filter((r) => r.nn === nq);
  const one = exact.length === 1 ? exact[0] : found.length === 1 ? found[0] : null;
  if (one) { go(one.id); return one.id; }
  openSearch(v);
  return null;
}
function closeCard() {
  card.hidden = true;
  announce(null);
  card.classList.remove("peek");
  current = null;
  trail = [];
  clearEgo();
  if (location.hash) history.pushState(null, "", location.pathname + location.search);
  const back = opener && doc.contains(opener) ? opener : $("#search-input");
  back.focus({ preventScroll: true });
}
$("#card-close").addEventListener("click", closeCard);
$("#card-back").addEventListener("click", () => history.back());
$("#card-showmap").addEventListener("click", () => {
  card.classList.add("peek");
  $("#map0").scrollIntoView({ block: "start", behavior: "auto" });
});
$(".card-head").addEventListener("click", (e) => {
  if (card.classList.contains("peek") && !e.target.closest("button")) card.classList.remove("peek");
});
doc.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !card.hidden) {
    if (!$("#map-pick").hidden) { hidePick(); return; }
    closeCard();
  }
});

function route() {
  const id = parseHash();
  if (id == null) { if (!card.hidden) { card.hidden = true; clearEgo(); current = null; announce(null); } return; }
  if (!MO.size) return; // данные ещё грузятся: route() вызовется после загрузки
  const r = MO.get(id);
  if (!r) { $("#card-status").textContent = "Муниципалитет не найден"; return; }
  if (current !== id) netOn = false;
  current = id;
  renderCard(r);
  card.hidden = false;
  card.classList.remove("peek");
  card.scrollTop = 0;
  $("#card-back").hidden = !trail.length;
  $("#card-title").focus({ preventScroll: true });
  $("#card-status").textContent = `Карточка: ${r.n}, ${r.r}`;
  drawEgo(r);
  announceCard(r);
}
function announceCard(r) {
  const nd = nodeOf(r) || r; // район столицы — ячейка города
  const ids = (xs) => (xs || []).map((p) => p[0]);
  if (netOn && nd.nb && nd.nb.length) announce(nd.hq != null ? nd.id : null, [], [], ids(nd.nb));
  else announce(nd.hq != null ? nd.id : null, ids(nd.sim), ids(nd.simb));
}

// --- карточка ------------------------------------------------------------------------------------------------
const nodeOf = (r) => (r.role === "inner" ? MO.get(r.node) : r); // район столицы -> узел-город
function simList(pairs, numbered) {
  if (!pairs || !pairs.length) return "";
  return `<ol class="sim ${numbered ? "numbered" : "plain"}">` + pairs.map(([id, km]) => {
    const o = MO.get(id) || { n: "№ " + id, r: "", t: null };
    return `<li><button type="button" data-go="${id}">${fig(o.t)}<span><span class="nm">${esc(cap1(o.ns || o.n))}</span><span class="rg">${esc(o.r)} — ${esc(typeName(o.t))}</span></span><span class="km">${nf.format(km)} км</span></button></li>`;
  }).join("") + "</ol>";
}
// полоса «почему»: штрихи квантилей всех узлов, межквартильный размах и медиана типа, точка МО (акцент)
function strip(w, v) {
  const q = (w.q || []).filter((x) => x != null);
  const vals = [...q, w.lo, w.hi, w.med, v].filter((x) => x != null);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const W = 300, H = 22, pad = 6;
  const x = (u) => pad + ((u - lo) / (hi - lo)) * (W - 2 * pad);
  let g = "";
  if (w.lo != null && w.hi != null) g += `<rect class="strip-iqr" x="${x(w.lo)}" y="3" width="${Math.max(1, x(w.hi) - x(w.lo))}" height="${H - 6}"/>`;
  for (const u of q) g += `<line class="strip-q" x1="${x(u)}" x2="${x(u)}" y1="7" y2="${H - 7}"/>`;
  if (lo < 0 && hi > 0) g += `<line class="strip-zero" x1="${x(0)}" x2="${x(0)}" y1="0" y2="${H}"/>`;
  if (w.med != null) g += `<line class="strip-med" x1="${x(w.med)}" x2="${x(w.med)}" y1="1" y2="${H - 1}"/>`;
  if (v != null) g += `<circle class="dot-mo" cx="${x(v)}" cy="${H / 2}" r="5"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(w.label)}: точка — муниципалитет, чёрная засечка — медиана типа, серая полоса — середина типа, штрихи — все муниципалитеты">${g}</svg>`;
}
function basketChart(b, t) {
  if (!b) return "";
  const ty = TYPES.get(t);
  const med = ty && ty.basket ? ty.basket.med : null;
  const unst = new Set((view.unstable_parts || []).filter((s) => s.startsWith(t + ":")).map((s) => s.split(":")[1].replace("clr_rel_", "")));
  const lim = Math.max(0.1, ...b.map((u) => Math.abs(u || 0)), ...(med || []).map((u) => Math.abs(u || 0))) * 1.05;
  const pc = (v) => 50 + (v / lim) * 50; // 0 — «как обычно в регионе» — посередине
  let h = '<div class="bars" role="img" aria-label="Корзина трат относительно своего региона: полоса вправо — доля части больше, чем обычно в регионе, влево — меньше">';
  PART_KEYS.forEach((p, i) => {
    const v = b[i] ?? 0, u = unst.has(p);
    const l = Math.min(pc(0), pc(v)), w = Math.max(0.6, Math.abs(pc(v) - pc(0)));
    h += `<div class="bar-row"><span class="bar-lab">${esc(PART_LABELS[p])}${u ? "<small>знак меняется*</small>" : ""}</span><span class="track">`;
    h += `<i class="bar${u ? " unst" : ""}" style="left:${l.toFixed(1)}%;width:${w.toFixed(1)}%"></i>`;
    if (med && med[i] != null) h += `<i class="tick" style="left:${pc(med[i]).toFixed(1)}%"></i>`;
    h += `</span></div>`;
  });
  // порция 6m (judge-c6): числовая шкала — «×0,67», «×1», «×1,5»: во сколько раз доля части отличается от региона
  const mult = [1.25, 1.5, 2, 3, 4].filter((m) => Math.log(m) <= lim).pop();
  if (mult) {
    const fx = (v) => "×" + String(Math.round(v * 100) / 100).replace(".", ",");
    h += `<div class="bar-row bar-scale" aria-hidden="true"><span class="bar-lab"></span><span class="track scale">`;
    h += [[-Math.log(mult), fx(1 / mult)], [0, "×1"], [Math.log(mult), fx(mult)]].map(([v, s]) => `<em style="left:${pc(v).toFixed(1)}%">${s}</em>`).join("");
    h += `</span></div>`;
  }
  // порция 6m: стрелки не отрываются от слов при переносе (неразрывный пробел)
  return h + '</div><div class="axis-words bar-axis"><span>← меньше, чем обычно в регионе</span><span>больше →</span></div>';
}
function pathChart(win, st) {
  if (!win) return "";
  const n = win.length, W = 300, c = W / n, H = 26;
  let g = `<defs><pattern id="phatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="4" stroke="#fff" stroke-width="2"/></pattern></defs>`;
  for (let i = 0; i < n; i++) {
    const t = win[i], s = st ? st[i] : "n";
    const pale = s === "w" || (s === "r" && T3 === "not");
    g += `<rect x="${i * c + 1}" y="2" width="${c - 2}" height="${H - 8}" fill="var(--t${t})" opacity="${pale ? 0.45 : 1}"/>`;
    if (pale) g += `<rect x="${i * c + 1}" y="2" width="${c - 2}" height="${H - 8}" fill="url(#phatch)"/>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Тип по 13 скользящим годовым окнам с января–декабря 2023 по январь–декабрь 2024; заштрихованы окна со сменой в пределах шума">${g}</svg>
    <div class="axis-words"><span>янв–дек 2023</span><span>13 скользящих окон по 12 месяцев</span><span>янв–дек 2024</span></div>`;
}
function rhythmChart(rh) {
  const W = 300, H = 60, pad = 4;
  const lim = Math.max(1, ...rh.map((v) => Math.abs(v || 0)));
  const x = (i) => pad + (i * (W - 2 * pad)) / 11, y = (v) => H / 2 - (v / lim) * (H / 2 - 4);
  const d = rh.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v || 0).toFixed(1)}`).join("");
  const g = `<line class="strip-zero" x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}"/><path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  return `<div class="rh"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Свой годовой ритм трат по месяцам относительно общего календарного ритма">${g}</svg><span class="rh-zero">общий ритм</span></div>
    <div class="axis-words">${MONTHS.map((m) => `<span>${m}</span>`).join("")}</div>`;
}
// 2023 → 2024: типы годовых окон и итоговый тип (по всем 24 месяцам) говорятся одной строкой, без путаницы
function statusLine(r) {
  if (r.t23 == null || r.t24 == null) return "";
  const T = (t) => `тип «${esc(typeName(t))}»`;  // порция 6a: название, а не номер
  if (r.t23 === r.t24) {
    if (r.t == null || r.t === r.t23) return `<p class="status">В 2023 и 2024 годах тип не менялся.</p>`;
    return `<p class="status">В 2023 и 2024 годах по отдельности — ${T(r.t23)}. Итоговый ${T(r.t)} посчитан по всем 24 месяцам сразу, поэтому может отличаться от типа отдельного года.</p>`;
  }
  const noise = T3 === "not" || !r.rel;
  let h = noise
    ? `<p class="status noise">${esc(CARD.status_noise || "Смена типа в пределах шума")}.</p>`
    : `<p class="status solid">${esc(CARD.status_reliable || "Смена типа: обе половины года согласны")}.</p>`;
  return h;
}
// устойчивость словами, одной строкой рядом с типом; варианты расчёта и повторы с другим seed — раздельно (§3.9)
// порция 6b: флаг устойчивости типа (этап usefulness, mo_flags.csv) — словом перед счётом; знак разности
// ошибок сверки у отдельного МО не показывается (usefulness.rule_share.per_mo: not_shown)
function stabilityLine(nd) {
  const rr = (s) => (s ? s.split("/").map(Number) : null);
  const a = rr(nd.rob_rule), b = rr(nd.rob_seed);
  if (!a && !b) return "";
  const flag = nd.fl ? (CARD.flag_words || {})[nd.fl] : "";
  const vars = (CARD.variants || {})[a ? String(a[1]) : ""] || "";
  // порция 6c: какие варианты дали другой тип — по nd.var (типы в вариантах R1, порядок — CARD.var_words);
  // все варианты дали тот же тип — перечисляются сами варианты отдельной фразой, а не в скобках у счёта
  const vw = CARD.var_words || [];
  const other = (nd.var || []).map((v, i) => [v, vw[i]]).filter(([v, w]) => v != null && w && v !== nd.t);
  const list = (xs) => (xs.length > 1 ? xs.slice(0, -1).join(", ") + " и " + xs[xs.length - 1] : xs[0] || "");
  const bits = [];
  if (a) {
    bits.push(`тот же тип — в ${a[0]} из ${a[1]} ${plural(a[1], "варианта", "вариантов", "вариантов")} расчёта`);
    if (other.length && other.length === a[1] - a[0]) {
      bits.push(`другой — ${other.length > 1 ? "в вариантах" : "в варианте"} ${list(other.map(([v, w]) => `«${esc(w)}» (тип «${esc(typeName(v))}»)`))}`);
    } else if (vars) bits.push(`варианты: ${esc(vars)}`);
  }
  if (b) bits.push(`тот же тип — в ${b[0]} из ${b[1]} ${plural(b[1], "повтора", "повторов", "повторов")} с другим случайным начальным значением`);
  // порция 6j: «Устойчивость типа: Тип устойчив.» — повтор; при флаге — только он
  const head = flag ? `<b class="flag-word flag-${esc(nd.fl[0])}">${esc(cap1(flag))}.</b> ` : `<b>${esc(CARD.stability || "Устойчивость типа")}:</b> `;
  const body = bits.map(cap1).join(". ");
  return `<p class="rob-line">${head}${body}. <span class="label-note">${esc(story.stability_label || "описание: заранее записанной проверки нет")}</span></p>`;
}

function renderCard(r) {
  const nd = nodeOf(r) || r;
  const ty = nd.t != null ? TYPES.get(nd.t) : null;
  // заголовок — короткое имя с прописной, полное официальное название — в подзаголовке рядом с регионом
  $("#card-title").textContent = cap1(r.ns || r.n);
  $("#card-sub").textContent = [r.r, r.n !== (r.ns || r.n) ? r.n : r.k].filter(Boolean).join(" · ");
  let h = "";

  // 1. Тип и устойчивость — рядом (порция 5b)
  h += `<section class="row" aria-labelledby="c-type"><h3 id="c-type">Тип</h3>`;
  if (r.role === "inner") h += `<p class="kv">Район входит в узел-город ${esc(nd.n)}; тип — у города целиком.</p>`;
  if (nd.t == null) {
    const nodata = (CH.explore || {}).nodata || "Нет типа: ряд трат неполный (меньше 24 месяцев или с пропусками)";
    const why = r.why_null ? `Нет типа: ${r.why_null}` : nodata; // причина этого МО точнее общей строки
    h += `<div class="type-line">${fig(null)}<span>Нет типа</span></div><p class="cap">${esc(why)}.</p>`;
  } else {
    h += `<div class="type-line">${typeHTML(nd.t)}</div><p class="cap">${esc(names.caption || "Относительно своего региона")}${CARD.type_note ? ". " + esc(CARD.type_note) + "." : ""}</p>`;
    h += stabilityLine(nd);
    if (r.role === "inner") h += `<p><button type="button" class="tool tool-text" data-go="${nd.id}">Карточка города</button></p>`;
  }
  h += `</section>`;

  // 2. С кем сверять изменения: соседи по своему региону (набор B T7), затем похожие по тратам в других регионах
  if (nd.simb && nd.simb.length) {
    // порция 6j: строка — как совет по размеру: у крупных (флаг lg; вне базы — по населению) своя;
    // порция 6l: у крупных и заголовок — про оба ориентира
    const big = nd.lg != null ? nd.lg === 1 : (nd.pop || r.pop || 0) >= (CARD.large_pop || Infinity);
    const st = (big && CARD.simb_title_large) || CARD.simb_title || "Соседи по своему региону";
    h += `<section class="row sim-b" aria-labelledby="c-simb"><h3 id="c-simb">${esc(st)}</h3>${simList(nd.simb, false)}`;
    const sn = (big && CARD.simb_note_large) || CARD.simb_note;
    if (sn) h += `<p class="sim-note">${esc(sn)}.</p>`;
    h += `</section>`;
  }
  if (nd.t != null) {
    h += `<section class="row sim-p" aria-labelledby="c-sim"><h3 id="c-sim">${esc(CARD.sim_title || cap1(view.set_name || "сопоставимые территории"))}</h3>`;
    if (nd.sim && nd.sim.length && CARD.sim_from) h += `<p class="sim-from">${esc(CARD.sim_from)}.</p>`;
    if (nd.sim && nd.sim.length) {
      h += simList(nd.sim, true);
      // порция 6f: крупный муниципалитет (верхняя пятая часть по населению, usefulness.size_posthoc) — свой
      // регион не единственный ориентир; знак разности ошибок у отдельного МО по-прежнему не показывается
      if (nd.lg && CARD.large_note) h += `<p class="sim-large">${esc(CARD.large_note)}.</p>`;
      const cc = CARD.sim_caption || (CH.comparable || {}).similar_caption;
      h += `<p class="cap">${CARD.sim_note ? esc(CARD.sim_note) + ". " : ""}${esc(CARD.lines || "Сходство корзин трат; поездки и потоки здесь не учитываются")}. ${esc(CARD.shifted || "")}.${cc ? " " + esc(cc) + "." : ""}</p>`;
    } else if (CARD.no_comparable) {
      h += `<p class="kv">${esc(CARD.no_comparable)}.</p>`;
    }
    h += `</section>`;
  }
  // порция 6e: соседи по основной сети корзин (та сеть, на которой построены типы) — переключатель: список
  // и дуги на карте вместо соседей по региону и похожих по тратам
  if (nd.t != null && nd.nb && nd.nb.length && CARD.net_title) {
    h += `<section class="row net" aria-labelledby="c-net"><div class="net-head"><h3 id="c-net">${esc(CARD.net_title)}</h3>`;
    h += `<button type="button" class="net-toggle" id="net-toggle" role="switch" aria-checked="${netOn}" aria-controls="net-body">${esc(CARD.net_toggle || "Показать списком и на карте")}</button></div>`;
    h += `<div id="net-body"${netOn ? "" : " hidden"}>${CARD.net_note ? `<p class="sim-note">${esc(CARD.net_note)}.</p>` : ""}${simList(nd.nb, false)}</div></section>`;
  }

  // Подробнее: 2023 → 2024, корзина, окна, «почему», соседи по региону, ритм, пометки
  const moreAt = h.length;
  h += `<details class="more"><summary>Подробнее</summary>`;
  const moreHead = h.length;
  if (nd.t != null && nd.t23 != null) {
    h += `<section class="row" aria-labelledby="c-path"><h3 id="c-path">Тип по годам</h3>`;
    // порция 6l (check-ux 04.10): тип по годам разный — тип выше посчитан по всем 24 месяцам
    if (nd.t23 !== nd.t24) h += `<p class="cap">Тип выше — по всем 24 месяцам; по годам отдельно он бывает другим.</p>`;
    h += `<dl class="path-line">`;
    h += `<dt>2023</dt><dd>${typeHTML(nd.t23)}</dd><dt>2024</dt><dd>${typeHTML(nd.t24)}</dd></dl>`;
    h += statusLine(nd) + `</section>`;
  }
  const bk = nd.b24 || nd.b23;
  if (bk) {
    h += `<section class="row basket" aria-labelledby="c-bk"><h3 id="c-bk">Корзина ${nd.b24 ? "2024" : "2023"} года относительно своего региона</h3>${basketChart(bk, nd.t)}<p class="cap">Серые полосы — муниципалитет, чёрная засечка — медиана типа.${CARD.basket_scale ? " " + esc(CARD.basket_scale) + "." : ""}${(view.unstable_parts || []).some((s) => s.startsWith(nd.t + ":")) ? " * Знак меняется: в одних вариантах расчёта медиана типа по этой части выше региона, в других — ниже." : ""}</p></section>`;
  }
  if (nd.win) {
    h += `<section class="row path" aria-labelledby="c-win"><h3 id="c-win">Тип по окнам</h3>${pathChart(nd.win, nd.st)}</section>`;
  }
  if (ty && ty.why && nd.why) {
    h += `<section class="row" aria-labelledby="c-why"><h3 id="c-why">Чем отличается тип</h3><p class="cap">Чем тип отличается от остальных; причины не проверяли</p><div class="why">`;
    ty.why.forEach((w, i) => {
      h += `<div class="why-row"><span class="lab">${esc(w.label)}<small>${w.kind === "place" ? "признак места" : "часть корзины"}</small></span>${strip(w, nd.why[i])}</div>`;
    });
    h += `</div><div class="axis-words" style="padding-left:calc(9.5em + 8px)"><span>← ниже</span><span>выше →</span></div>`;
    // порция 6c: полосы — признаки, которые выбрало правило названия (2 части корзины и 1 признак места);
    // меньше полос — остальные отличаются слабее порога правила (|δ Клиффа|), их не дорисовываем
    h += `<p class="cap">Признаки — по правилу названия типа: части корзины и признак места, которыми медиана типа сильнее всего отличается от остальных муниципалитетов.`;
    if (ty.why_n && ty.why.length < ty.why_n && ty.why_cliff != null) {
      const k = ty.why.length, W = { 1: "полоса одна", 2: "полосы две" };
      h += ` Остальные у этого типа отличаются слабее порога правила (размер отличия — дельта Клиффа — меньше ${String(ty.why_cliff).replace(".", ",")} по модулю), поэтому ${W[k] || "полос " + nf.format(k)} из ${ty.why_n === 3 ? "трёх" : nf.format(ty.why_n)}.`;
    }
    h += `</p></section>`;
  }
  if (nd.t != null && nd.second != null) {
    h += `<p class="kv">Ближе к центру типа ${typeHTML(nd.second)}, чем к центру своего</p>`;
  }
  if (nd.t != null) {
    h += `<section class="row rhythm" aria-labelledby="c-rh"><h3 id="c-rh">Свой годовой ритм</h3>`;
    h += nd.rh ? rhythmChart(nd.rh) + `<p class="cap">Отклонение от общего календарного ритма, повторилось в 2023 и 2024 годах.</p>` : `<p class="kv">Свой ритм не отличим от шума</p>`;
    h += `</section>`;
  }
  const flags = [];
  if (r.role === "city") flags.push(`Узел-город: ${nf.format(CITY_N.get(r.id) || 0)} районов считаются одним муниципалитетом`);
  if (r.wp) flags.push("Зарплаты и НДФЛ считаются по месту работы");
  if (r.ser && r.ser !== "full" && nd.t != null) flags.push("Ряд трат неполный");
  if (flags.length) h += `<section class="row" aria-labelledby="c-fl"><h3 id="c-fl">Пометки</h3><ul class="flags">${flags.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></section>`;
  h = h.length === moreHead ? h.slice(0, moreAt) : h + "</details>";
  h += `<p class="src">${SRC}</p>`;
  cardBody.innerHTML = nbsp(h);
}
cardBody.addEventListener("click", (e) => {
  const sw = e.target.closest("#net-toggle");
  if (sw) {
    netOn = !netOn;
    sw.setAttribute("aria-checked", String(netOn));
    const body = $("#net-body");
    if (body) body.hidden = !netOn;
    const r = MO.get(current);
    if (r) { drawEgo(r); announceCard(r); }
    return;
  }
  const b = e.target.closest("[data-go]");
  if (b) go(Number(b.dataset.go), true);
});

// --- карта ячеек -----------------------------------------------------------------------------------------------
const svg = $("#hexmap");
const frame = $("#map-frame");
const tip = $("#map-tip");
const pick = $("#map-pick");
let base = null, zoom = 1, center = null;
const SQ3 = Math.sqrt(3);
function hexCenter(q, r) { return [HEX.x0 + HEX.size * SQ3 * (q + r / 2), HEX.y0 - HEX.size * 1.5 * r]; }
function hexPath(q, r, k = 1) {
  const [cx, cy] = hexCenter(q, r), s = HEX.size * k;
  const pts = [[0, -s], [SQ3 / 2 * s, -s / 2], [SQ3 / 2 * s, s / 2], [0, s], [-SQ3 / 2 * s, s / 2], [-SQ3 / 2 * s, -s / 2]];
  return "M" + pts.map(([a, b]) => `${(cx + a).toFixed(1)},${(cy + b).toFixed(1)}`).join("L") + "Z";
}
function toSvg(ev) {
  const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}
function cellAt(pt) {
  const rx = pt.x - HEX.x0, ry = -(pt.y - HEX.y0);
  const r = ry / (1.5 * HEX.size), q = rx / (SQ3 * HEX.size) - r / 2;
  let x = q, z = r, y = -x - z;
  let rx2 = Math.round(x), ry2 = Math.round(y), rz2 = Math.round(z);
  const dx = Math.abs(rx2 - x), dy = Math.abs(ry2 - y), dz = Math.abs(rz2 - z);
  if (dx > dy && dx > dz) rx2 = -ry2 - rz2; else if (dy <= dz) rz2 = -rx2 - ry2;
  return CELLS.get(rx2 + "," + rz2) || null;
}
const pxPerUnit = () => { const m = svg.getScreenCTM(); return m && m.a ? m.a : 0.1; };
// подписи и номера на карте — в экранных пикселях при любом масштабе: --u = единиц viewBox на пиксель
function scaleLabels() { svg.style.setProperty("--u", (1 / pxPerUnit()).toFixed(3)); }
function cellLabel(c) {
  if (c.role === "city") return `${cap1(c.ns || c.n)} — город целиком, ${nf.format(CITY_N.get(c.id) || 0)} районов — ${typeName(c.t)}`;
  return `${cap1(c.ns || c.n)} — ${c.r} — ${typeName(c.t)}`;
}
function showTip(c, ev) {
  const fr = frame.getBoundingClientRect();
  tip.innerHTML = nbsp(`${fig(c.t)} <b>${esc(cap1(c.ns || c.n))}</b><br><span class="tip-meta">${esc(c.role === "city" ? "город целиком, " + nf.format(CITY_N.get(c.id) || 0) + " районов" : c.r)} — ${esc(typeName(c.t))}</span>`);
  tip.hidden = false;
  const x = ev.clientX - fr.left, y = ev.clientY - fr.top;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(0, Math.min(fr.width - w, x + 14)) + "px";
  tip.style.top = (y - h - 12 < 0 ? y + 16 : y - h - 12) + "px";
}
function hideTip() { tip.hidden = true; const hv = svg.querySelector(".hover"); if (hv) hv.remove(); }
function hoverCell(c) {
  let hv = svg.querySelector(".hover");
  if (!hv) { hv = doc.createElementNS(SVGNS, "path"); hv.setAttribute("class", "hover"); svg.appendChild(hv); }
  hv.setAttribute("d", hexPath(c.hq, c.hr));
}
function hidePick() { pick.hidden = true; pick.innerHTML = ""; }
function showPick(cands, ev) {
  const fr = frame.getBoundingClientRect();
  pick.innerHTML = `<p>Рядом несколько муниципалитетов:</p>` + cands.map((c) => `<button type="button" data-go="${c.id}">${fig(c.t)} ${esc(cellLabel(c))}</button>`).join("");
  pick.hidden = false;
  const x = ev.clientX - fr.left, y = ev.clientY - fr.top;
  pick.style.left = Math.max(0, Math.min(fr.width - pick.offsetWidth, x - pick.offsetWidth / 2)) + "px";
  pick.style.top = Math.min(fr.height - pick.offsetHeight, y + 12) + "px";
  pick.querySelector("button").focus();
}
pick.addEventListener("click", (e) => {
  const b = e.target.closest("[data-go]");
  if (b) { hidePick(); go(Number(b.dataset.go)); }
});
function setView() {
  const w = base[2] / zoom, h = base[3] / zoom;
  const x = Math.max(base[0], Math.min(base[0] + base[2] - w, center[0] - w / 2));
  const y = Math.max(base[1], Math.min(base[1] + base[3] - h, center[1] - h / 2));
  center = [x + w / 2, y + h / 2];
  svg.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
  svg.classList.toggle("zoomed", zoom > 1);
  scaleLabels();
  if (current != null) drawEgo(MO.get(current));
}
function initMap(hexgrid) {
  if (!svg) return;
  const ds = svg.dataset;
  const g = (hexgrid && hexgrid.grid) || {};
  HEX = { size: +(ds.hexSize ?? g.size), x0: +(ds.hexX0 ?? g.x0), y0: +(ds.hexY0 ?? g.y0) };
  if (!(HEX.size > 0)) { HEX = null; return; }
  base = svg.getAttribute("viewBox").split(/[\s,]+/).map(Number);
  center = [base[0] + base[2] / 2, base[1] + base[3] / 2];
  $("#map-tools").hidden = false;
  scaleLabels();
  let rz = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(rz);
    rz = requestAnimationFrame(() => { scaleLabels(); if (current != null) drawEgo(MO.get(current)); });
  }).observe(svg);
  const summary = [S0.scope, "Цвет и фигура — тип"].filter(Boolean).join(". ");
  svg.setAttribute("aria-label", summary);
  svg.removeAttribute("aria-labelledby");

  let down = null, moved = false;
  svg.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY, c: [...center] }; moved = false;
    if (zoom > 1 && e.pointerType !== "touch") svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener("pointermove", (e) => {
    if (down && zoom > 1 && (e.buttons || e.pointerType === "touch")) {
      const k = pxPerUnit();
      const dx = (e.clientX - down.x) / k, dy = (e.clientY - down.y) / k;
      if (Math.abs(dx) * k + Math.abs(dy) * k > 4) { moved = true; center = [down.c[0] - dx, down.c[1] - dy]; setView(); }
      return;
    }
    if (e.pointerType !== "mouse") return;
    const c = cellAt(toSvg(e));
    if (c) { hoverCell(c); showTip(c, e); svg.style.cursor = "pointer"; } else { hideTip(); svg.style.cursor = ""; }
  });
  svg.addEventListener("pointerleave", () => { if (!down) hideTip(); });
  svg.addEventListener("pointerup", (e) => {
    const wasDown = down; down = null;
    if (!wasDown || moved) return;
    const pt = toSvg(e);
    if (e.pointerType === "mouse") {
      const c = cellAt(pt);
      if (c) go(c.id);
      return;
    }
    // касание: ближайшие ячейки в радиусе 24 px; несколько — список из трёх
    const rad = 24 / pxPerUnit();
    const near = [];
    for (const c of CELLS.values()) {
      const [cx, cy] = hexCenter(c.hq, c.hr);
      const d = Math.hypot(cx - pt.x, cy - pt.y);
      if (d <= rad) near.push([d, c]);
    }
    near.sort((a, b) => a[0] - b[0]);
    if (!near.length) { hidePick(); return; }
    if (near.length === 1 || near[0][0] < HEX.size * 0.5) {
      hidePick(); showTip(near[0][1], e); go(near[0][1].id);
    } else showPick(near.slice(0, 3).map((x) => x[1]), e);
  });
  $("#map-tools").addEventListener("click", (e) => {
    const b = e.target.closest("[data-zoom]");
    if (!b) return;
    const a = b.dataset.zoom;
    if (a === "in") zoom = Math.min(8, zoom * 2);
    else if (a === "out") zoom = Math.max(1, zoom / 2);
    else { zoom = 1; center = [base[0] + base[2] / 2, base[1] + base[3] / 2]; }
    if (a === "in" && current != null) { const c = egoCell(MO.get(current)); if (c) center = hexCenter(c.hq, c.hr); }
    setView();
  });
  doc.addEventListener("click", (e) => { if (!pick.hidden && !pick.contains(e.target) && !svg.contains(e.target)) hidePick(); });
}
function egoCell(r) { const nd = r && nodeOf(r); return nd && nd.hq != null ? nd : null; }
function clearEgo() {
  const g = svg && svg.querySelector(".ego");
  if (g) g.remove();
  if (svg) svg.classList.remove("has-ego");
  const note = $("#ego-note");
  if (note) note.hidden = true;
}
// эго-сеть выбранного МО: акцентная обводка, прямые линии к сопоставимым территориям и их номера
function drawEgo(r) {
  if (!svg || !HEX) return;
  clearEgo();
  const nd = egoCell(r);
  if (!nd) return;
  const g = doc.createElementNS(SVGNS, "g");
  g.setAttribute("class", "ego");
  g.setAttribute("aria-hidden", "true");
  const [sx, sy] = hexCenter(nd.hq, nd.hr);
  const k = pxPerUnit() || 0.1;
  let s = "";
  const net = netOn && nd.nb && nd.nb.length;
  const sim = net ? [] : (nd.sim || []).map(([id]) => MO.get(id)).filter((o) => o && o.hq != null);
  // порция 6e: при переключателе «Соседи по сети корзин» — сплошные линии к соседям по сети, без номеров
  const simb = ((net ? nd.nb : nd.simb) || []).map(([id]) => MO.get(id)).filter((o) => o && o.hq != null);
  // похожие по тратам в других регионах — тонкий пунктир; соседи по своему региону — сплошные (порция 5b);
  // белая подложка под линией: без неё тёмная линия теряется на тёмных ячейках
  for (const o of sim) {
    const [x, y] = hexCenter(o.hq, o.hr);
    s += `<line class="casing thin" x1="${sx}" y1="${sy}" x2="${x}" y2="${y}"/><line class="far" x1="${sx}" y1="${sy}" x2="${x}" y2="${y}"/>`;
  }
  for (const o of simb) {
    const [x, y] = hexCenter(o.hq, o.hr);
    s += `<line class="casing" x1="${sx}" y1="${sy}" x2="${x}" y2="${y}"/><line x1="${sx}" y1="${sy}" x2="${x}" y2="${y}"/>`;
  }
  // порция 6m (judge-c6): остальные ячейки приглушены (CSS .has-ego) — выбранное МО и его списки — в цвете типа
  for (const o of [nd, ...sim, ...simb]) { const t = (nodeOf(o) || o).t; s += `<path class="lit" d="${hexPath(o.hq, o.hr)}" style="fill:${t != null ? `var(--t${t})` : "#d3d7dc"}"/>`; }
  for (const o of sim) s += `<path class="nbr" d="${hexPath(o.hq, o.hr)}"/>`;
  s += `<path class="sel-halo" d="${hexPath(nd.hq, nd.hr, 1.35)}"/><path class="sel" d="${hexPath(nd.hq, nd.hr, 1.35)}"/>`;
  const rad = 8 / k;
  sim.forEach((o, i) => {
    const [x, y] = hexCenter(o.hq, o.hr);
    s += `<g class="num"><circle cx="${x + HEX.size * 1.1}" cy="${y - HEX.size * 1.1}" r="${rad}"/><text x="${x + HEX.size * 1.1}" y="${y - HEX.size * 1.1}">${i + 1}</text></g>`;
  });
  g.innerHTML = s;
  const labels = svg.querySelector(".labels");
  svg.insertBefore(g, labels || null);
  svg.classList.add("has-ego"); // выноски примеров прячутся, пока на карте эго-сеть
  const note = $("#ego-note");
  if (note && net && simb.length) {
    note.innerHTML = nbsp(`<b>${esc(cap1(nd.ns || nd.n))}</b>: линии — ${esc((CARD.net_title || "").toLowerCase())}. ${esc(CARD.shifted || "")}.`);
    note.hidden = false;
  } else if (note && sim.length) {
    note.innerHTML = nbsp(`<b>${esc(cap1(nd.ns || nd.n))}</b>: сплошные линии — соседи по своему региону, пунктир с номерами 1–${sim.length} — ${esc((CARD.sim_title || view.set_name || "").toLowerCase())}. Линии — ${esc((CARD.lines || "").replace(/^С/, "с"))}. ${esc(CARD.shifted || "")}.`);
    note.hidden = false;
  }
}

// ключ типов под картой (на телефоне выноски на карте нечитаемы и скрыты)
function renderKey() {
  const ul = $("#map-key");
  if (ul.children.length) return; // вписан при сборке
  const order = view.legend_order || [...TYPES.keys()];
  const nd = (CH.explore || {}).nodata ? "нет типа" : "нет типа";
  ul.innerHTML = order.map((t) => {
    const ty = TYPES.get(t);
    return `<li>${fig(t)}<span>${esc(typeName(t))}${ty && ty.size ? ` · ${nf.format(ty.size)}` : ""}</span></li>`;
  }).join("") + `<li>${fig(null)}<span>${nd}</span></li>`;
}

// кнопки-примеры: story.screen0.examples (правило examples в landing.py) или первый типичный пример типа
// порция 6i: ссылка «Найти свой муниципалитет» на телефоне — прокрутка к поиску и фокус в поле
function bindFindLink() {
  const a = $("#hero-find"), inp = $("#search-input");
  if (!a || !inp) return;
  a.addEventListener("click", (e) => {
    e.preventDefault();
    // порция 6j: поле — у верха экрана (под шапкой), чтобы его не закрыла клавиатура
    const head = $(".masthead"), off = (head ? head.getBoundingClientRect().height : 56) + 12;
    const lab = $("#find") || inp;
    window.scrollTo({ top: lab.getBoundingClientRect().top + window.scrollY - off, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    inp.focus({ preventScroll: true });
  });
}

function renderExamples() {
  const box = $("#examples");
  if (box.children.length) { box.addEventListener("click", (e) => { const b = e.target.closest("[data-go]"); if (b) { e.preventDefault(); go(Number(b.dataset.go)); } }); return; }
  let ids = Array.isArray(S0.examples) ? S0.examples : [];
  if (!ids.length) {
    for (const t of view.legend_order || [...TYPES.keys()]) {
      const ty = TYPES.get(t);
      const e = ty && ty.examples && ty.examples.typical && ty.examples.typical[0];
      if (e != null) ids.push(e);
    }
    ids = ids.slice(0, 3);
  }
  box.innerHTML = ids.map((id) => MO.get(id)).filter(Boolean).map((r) =>
    `<button type="button" class="ex" data-go="${r.id}">${fig(r.t)}<span>${esc(r.ns || r.n)}<br><small>${esc(r.r)}</small></span></button>`).join("");
  box.addEventListener("click", (e) => { const b = e.target.closest("[data-go]"); if (b) go(Number(b.dataset.go)); });
}

// --- главы 1, 3, 4, 5: SVG и тексты вписаны при сборке (site_chapters.py); здесь только поведение ------------------
// ссылки на муниципалитеты в главах и в блоке «Что устояло» (пример пользы, порция 6b): go() запоминает,
// откуда открыли карточку (фокус вернётся туда)
for (const sec of doc.querySelectorAll(".chapter, #findings")) {
  sec.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-go], button[data-go]");
    if (!a) return;
    e.preventDefault();
    go(Number(a.dataset.go));
    // пример паспорта типа: карточка и карта (первый экран) — вместе
    if (a.dataset.map) $("#map0").scrollIntoView({ block: "start", behavior: "auto" });
  });
}
// глава 1: шаги переключают пример муниципалитета
{
  const fig = $("#basket-fig");
  if (fig) fig.addEventListener("click", (e) => {
    const b = e.target.closest("button.step");
    if (!b) return;
    for (const x of fig.querySelectorAll("button.step")) x.setAttribute("aria-pressed", String(x === b));
    for (const p of fig.querySelectorAll(".step-pane")) p.hidden = p.dataset.step !== b.dataset.step;
  });
}
// глава 5: поток или строка списка раскрывает муниципалитеты потока (тип окна 2023 -> тип окна 2024)
function flowMembers(a, b) {
  const out = [];
  for (const r of MO.values()) {
    if ((r.role === "territorial" || r.role === "city") && r.t23 === a && r.t24 === b) out.push(r);
  }
  return out.sort((x, y) => String(x.ns || x.n).localeCompare(String(y.ns || y.n), "ru"));
}
function showFlow(a, b) {
  const box = $("#flow-mo");
  if (!box) return;
  const key = a + "-" + b;
  const same = box.dataset.key === key && !box.hidden;
  for (const x of doc.querySelectorAll(".flow-btn")) x.setAttribute("aria-expanded", String(!same && x.dataset.a == a && x.dataset.b == b));
  for (const g of doc.querySelectorAll(".alluvial .chg")) g.classList.toggle("on", !same && g.dataset.a == a && g.dataset.b == b);
  if (same) { box.hidden = true; box.dataset.key = ""; return; }
  const rows = flowMembers(a, b);
  box.dataset.key = key;
  box.hidden = false;
  const head = `${fig(a)}${esc(typeName(a))} → ${fig(b)}${esc(typeName(b))}: ${nf.format(rows.length)} ${plural(rows.length, "муниципалитет", "муниципалитета", "муниципалитетов")}`;
  box.innerHTML = `<p><b>${head}</b></p><ul>${rows.map((r) =>
    `<li><a href="#mo=${r.id}" data-go="${r.id}">${esc(r.ns || r.n)}</a> <small>${esc(r.r)}</small></li>`).join("")}</ul>`;
}
{
  const dyn = $("#dynamics");
  if (dyn) {
    dyn.addEventListener("click", (e) => {
      const t = e.target.closest(".flow-btn, .alluvial .chg");
      if (t) showFlow(Number(t.dataset.a), Number(t.dataset.b));
    });
    dyn.addEventListener("keydown", (e) => {
      const t = e.target.closest(".alluvial .chg");
      if (t && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); showFlow(Number(t.dataset.a), Number(t.dataset.b)); }
    });
  }
}

// глава 9: сортировка таблиц «метод × индекс» по заголовку столбца (значения — в data-v ячеек)
for (const table of doc.querySelectorAll("table.mtable")) {
  table.addEventListener("click", (e) => {
    const b = e.target.closest("button.sort");
    if (!b) return;
    const th = b.parentElement;
    const col = Number(b.dataset.col);
    const num = th.dataset.kind === "num";
    const dir = th.getAttribute("aria-sort") === "descending" ? "ascending" : "descending";
    for (const x of table.querySelectorAll("th")) x.removeAttribute("aria-sort");
    th.setAttribute("aria-sort", dir);
    const sign = dir === "ascending" ? 1 : -1;
    const body = table.tBodies[0];
    const key = (tr) => tr.cells[col].dataset.v ?? tr.cells[col].textContent.trim();
    const rows = [...body.rows].sort((x, y) => {
      const a = key(x), c = key(y);
      if (a === "" || c === "") return (a === "") - (c === "");
      return sign * (num ? Number(a) - Number(c) : a.localeCompare(c, "ru"));
    });
    body.append(...rows);
  });
}

// --- таблица всех муниципалитетов (порция 6c): текстовая альтернатива карте --------------------------------------
// Строки — из mo.json, страницами по 50 (2192 строки сразу в DOM не идут); фильтры по региону, типу и началу
// названия (та же нормализация, что у поиска); сортировка по заголовку с aria-sort; название открывает карточку.
const ALL = { rows: null, view: [], page: 0, col: 0, dir: 1, size: 50 };
const FLAG_ORDER = { s: 0, d: 1 };
function allRows() {
  const order = view.legend_order || [...TYPES.keys()];
  const rank = (t) => (t == null ? 99 : order.indexOf(t));
  const out = [];
  for (const r of MO.values()) {
    const nd = nodeOf(r) || r;
    const name = cap1(r.ns || r.n) + (r.role === "city" ? " (город целиком)" : "");
    const nn = norm(r.n), ns = norm(r.ns || r.n);
    out.push({
      id: r.id, name, region: r.r || "", t: nd.t ?? null, fl: nd.fl || null, t23: nd.t23 ?? null, t24: nd.t24 ?? null,
      inner: r.role === "inner", nn, ns, words: (nn + " " + ns).split(" "),
      key: [name, r.r || "", rank(nd.t ?? null), nd.fl ? FLAG_ORDER[nd.fl] ?? 5 : 9, rank(nd.t23 ?? null), rank(nd.t24 ?? null)],
    });
  }
  return out;
}
function allFilter() {
  const q0 = norm($("#mo-q").value), reg = $("#mo-region").value, ty = $("#mo-type").value;
  const pick = (q) => ALL.rows.filter((x) =>
    (!reg || x.region === reg) &&
    (ty === "" || (ty === "0" ? x.t == null : x.t === Number(ty))) &&
    (!q || x.ns.startsWith(q) || x.nn.startsWith(q) || x.words.some((w) => w.startsWith(q))));
  ALL.view = pick(q0);
  // как в поиске: «Кириши» -> «Киришский» — без одной-двух последних букв, не короче 4
  for (const cut of [1, 2]) if (!ALL.view.length && q0.length >= 5 && q0.length - cut >= 4) ALL.view = pick(q0.slice(0, -cut));
  const c = ALL.col, s = ALL.dir;
  ALL.view.sort((a, b) => {
    const u = a.key[c], v = b.key[c];
    const d = typeof u === "number" ? u - v : String(u).localeCompare(String(v), "ru");
    return s * d || a.name.localeCompare(b.name, "ru") || a.region.localeCompare(b.region, "ru");
  });
  ALL.page = 0;
  allRender();
}
function allRender() {
  const n = ALL.view.length, a = ALL.page * ALL.size, b = Math.min(n, a + ALL.size);
  const heads = [...$("#mo-table").tHead.rows[0].cells].map((th) => th.textContent.trim());
  const yt = (t) => (t == null ? "—" : `${fig(t)}<span>${esc(typeName(t))}</span>`);
  const fw = CARD.flag_words || {};
  $("#mo-table").tBodies[0].innerHTML = nbsp(ALL.view.slice(a, b).map((x) =>
    `<tr><th scope="row" data-l="${esc(heads[0])}"><button type="button" class="mo-open" data-go="${x.id}">${esc(x.name)}</button></th>` +
    `<td data-l="${esc(heads[1])}">${esc(x.region)}</td>` +
    `<td data-l="${esc(heads[2])}">${x.t == null ? fig(null) + "<span>нет типа</span>" : yt(x.t)}${x.inner ? "<small>тип города целиком</small>" : ""}</td>` +
    `<td data-l="${esc(heads[3])}">${x.fl && fw[x.fl] ? esc(cap1(fw[x.fl])) : "—"}</td>` +
    `<td data-l="${esc(heads[4])}">${yt(x.t23)}</td><td data-l="${esc(heads[5])}">${yt(x.t24)}</td></tr>`).join(""));
  $("#mo-status").textContent = n
    ? `Строки ${nf.format(a + 1)}–${nf.format(b)} из ${nf.format(n)}${n < ALL.rows.length ? ` (всего ${nf.format(ALL.rows.length)})` : ""}`
    : "Ничего не нашлось: измените фильтр";
  $("#mo-prev").disabled = ALL.page === 0;
  $("#mo-next").disabled = b >= n;
  $("#mo-pager").hidden = n <= ALL.size;
}
function initAll() {
  const sec = $("#all-mo");
  if (!sec || ALL.rows || !MO.size) return;
  ALL.rows = allRows();
  const regs = [...new Set(ALL.rows.map((x) => x.region))].filter(Boolean).sort((a, b) => a.localeCompare(b, "ru"));
  $("#mo-region").insertAdjacentHTML("beforeend", regs.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join(""));
  const order = view.legend_order || [...TYPES.keys()];
  $("#mo-type").insertAdjacentHTML("beforeend", order.map((t) => `<option value="${t}">${SHAPES[t] || ""} ${esc(typeName(t))}</option>`).join("") + `<option value="0">Нет типа</option>`);
  for (const id of ["mo-ctl", "mo-wrap"]) $("#" + id).hidden = false;
  let tq = 0;
  $("#mo-q").addEventListener("input", () => { clearTimeout(tq); tq = setTimeout(allFilter, 120); });
  $("#mo-region").addEventListener("change", allFilter);
  $("#mo-type").addEventListener("change", allFilter);
  $("#mo-prev").addEventListener("click", () => { ALL.page = Math.max(0, ALL.page - 1); allRender(); $("#mo-table").scrollIntoView({ block: "start" }); });
  $("#mo-next").addEventListener("click", () => { ALL.page += 1; allRender(); $("#mo-table").scrollIntoView({ block: "start" }); });
  const table = $("#mo-table");
  table.tHead.addEventListener("click", (e) => {
    const b = e.target.closest("button.sort");
    if (!b) return;
    const th = b.parentElement, col = Number(b.dataset.col);
    const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
    for (const x of table.tHead.rows[0].cells) x.removeAttribute("aria-sort");
    th.setAttribute("aria-sort", dir);
    ALL.col = col; ALL.dir = dir === "ascending" ? 1 : -1;
    allFilter();
  });
  // название открывает карточку: обработчик ссылок глав (.chapter, button[data-go]) выше
  allFilter();
}
// --- загрузка ----------------------------------------------------------------------------------------------------
(async () => {
  const [mo, types, hexgrid] = await Promise.all([loadData("mo"), loadData("types"), loadData("hexgrid")]);
  for (const t of rowsOf(types && types.types ? types.types : types)) TYPES.set(Number(t.t ?? t.type ?? t.number), t);
  for (const r of rowsOf(mo)) {
    MO.set(r.id, r);
    if (r.role === "inner") CITY_N.set(r.node, (CITY_N.get(r.node) || 0) + 1);
  }
  initMap(hexgrid);
  // ячейка — у каждого узла сети и у каждого МО без типа (штриховка тоже открывает карточку с причиной)
  if (HEX) for (const r of MO.values()) if (r.hq != null && r.role !== "inner") CELLS.set(r.hq + "," + r.hr, r);
  renderKey();
  renderExamples();
  bindFindLink();
  openParam();
  initAll(); // строк в DOM — только страница из 50, индекс 2192 записей строится за миллисекунды
  if (!MO.size) {
    $("#search-hint").textContent += " Карточки недоступны: данные не загрузились.";
  }
  route();
})();
