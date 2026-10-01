// Объёмная карта первого экрана «Корзина и регион» (three.js 0.170.0, MIT, из vendor/ через import map).
// Ничего не считает: положения ячеек на карте и в островах, высоты (доля кафе относительно региона, exp CLR)
// и подписи островов готовит этап site (src/munnet/site_scene.py, data/scene.json). Здесь — только рисование,
// камера и форматирование чисел. Без WebGL, без данных сцены или при ошибке загрузки three.js первый экран
// остаётся статичной SVG-картой (landing.js), а карточка МО — та же, что открывает поиск.

const doc = document;
const $ = (s) => doc.querySelector(s);
const NB = " ";
const params = new URLSearchParams(location.search);
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches || (params.has("still") && params.get("still") !== "0");
const coarse = matchMedia("(pointer: coarse)").matches;

function readJSON(id) {
  const el = doc.getElementById(id);
  if (!el || !el.textContent.trim()) return null;
  try { return JSON.parse(el.textContent); } catch (e) { return null; }
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const nbsp = (s) => String(s ?? "").replace(/(^|[\s>(«])([вксоуиаяВКСОУИАЯ]) /g, "$1$2" + NB).replace(/ — /g, NB + "— ");
const cap1 = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const fmt1 = (v) => v.toFixed(1).replace(".", ",");
const nf = new Intl.NumberFormat("ru-RU");

// выбор, пришедший до готовности сцены (карточка открыта по адресу #mo=…), применяется после загрузки
let pending = null;
let onSelect = (id, sim, simb) => { pending = [id, sim, simb]; };
doc.addEventListener("munnet:select", (e) => onSelect(e.detail ? e.detail.id : null, (e.detail && e.detail.sim) || [], (e.detail && e.detail.simb) || []));

async function loadScene() {
  const inl = readJSON("data-scene");
  if (inl) return inl;
  if (location.protocol === "file:") return null;
  try { const r = await fetch("data/scene.json"); return r.ok ? await r.json() : null; } catch (e) { return null; }
}
function webgl() {
  try {
    const c = doc.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (e) { return false; }
}

async function main() {
  const canvas = $("#scene"), stage = $("#map0");
  if (!canvas || !stage || !webgl()) return;
  const scene0 = await loadScene();
  if (!scene0 || !scene0.cells) return;
  const THREE = await import("three");
  const { OrbitControls } = await import("./vendor/OrbitControls.js");

  const story = readJSON("story") || {};
  const view = story.view || {};
  const HX = (story.screen0 || {}).hero || {};
  const TYPE_NAMES = (story.names || {}).final || {};
  const PAL = Object.assign({ 0: "#d3d7dc" }, view.type_colors || {});
  const typeName = (t) => (t ? TYPE_NAMES[t] || `Тип ${t}` : HX.untyped || "Без типа");

  // названия и регионы — из встроенного индекса поиска (id, n, r — номер в rl, t)
  const idx = readJSON("names") || {};
  const RL = idx.rl || [];
  const NAME = new Map();
  (idx.id || []).forEach((id, j) => NAME.set(id, { n: idx.n[j], r: typeof idx.r[j] === "number" ? RL[idx.r[j]] : idx.r[j] }));

  const G = scene0.grid, C = scene0.cells, S = G.size, UNIT = scene0.unit_h || 22;
  const W2 = G.width / 2, H2 = G.height / 2;
  const cells = C.id.map((id, i) => {
    const nm = NAME.get(id) || { n: "№ " + id, r: "" };
    const ratio = C.h[i];
    return {
      id, t: C.t[i], ratio, name: nm.n, region: nm.r, fl: C.f ? C.f[i] : null,
      x: C.x[i] - W2, z: C.y[i] - H2, ix: C.ix[i] - W2, iz: C.iy[i] - H2,
      h: ratio == null ? 1.5 : UNIT * ratio,
    };
  });
  const N = cells.length;
  const byId = new Map(cells.map((c, i) => [c.id, i]));
  // подписи крупнейших городов (порция 6a): те же, что на плоской карте (hexgrid.json, cities — по населению,
  // считает этап site); HTML поверх холста, перекрывающиеся прячутся, при выборе МО и в островах не видны
  let hexg = readJSON("data-hexgrid");
  if (!hexg && location.protocol !== "file:") {
    try { const r = await fetch("data/hexgrid.json"); hexg = r.ok ? await r.json() : null; } catch (e) { hexg = null; }
  }
  const CITIES = ((hexg || {}).cities || []).map((c) => ({ i: byId.get(c.id), name: c.name })).filter((c) => c.i !== undefined);
  const islands = (scene0.islands || []).map((s) => ({ ...s, cx: s.x - W2, cz: s.y - H2 }));

  function ratioText(r) {
    if (r == null) return `${HX.untyped || "Без типа"}: ${HX.untyped_note || ""}`;
    if (Math.abs(Math.log(r)) >= 0.4) return `доля кафе примерно в${NB}${fmt1(r > 1 ? r : 1 / r)} раза ${r > 1 ? "больше" : "меньше"}, чем в${NB}регионе`;
    const p = Math.round((r - 1) * 100);
    return p === 0 ? `доля кафе как в${NB}регионе` : `доля кафе примерно на${NB}${Math.abs(p)}% ${p > 0 ? "больше" : "меньше"}, чем в${NB}регионе`;
  }

  // ------------------------------------------------------------------ сцена
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (e) { return; }
  doc.documentElement.classList.add("has-3d");
  canvas.hidden = false;
  $("#h0-ui").hidden = false;
  $("#h0-actions").hidden = false;
  const howto = $("#h0-howto");
  if (coarse && howto && howto.dataset.touch) howto.textContent = howto.dataset.touch;

  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 1, 20, 8000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8e98a6, 1.55));
  const sun = new THREE.DirectionalLight(0xffffff, 2.1);
  sun.position.set(-380, 720, -260);
  sun.castShadow = true;
  sun.shadow.mapSize.set(coarse ? 1024 : 2048, coarse ? 1024 : 2048);
  Object.assign(sun.shadow.camera, { left: -1000, right: 1000, top: 700, bottom: -700, near: 50, far: 2600 });
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(7000, 7000), new THREE.ShadowMaterial({ opacity: 0.17 }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const geo = new THREE.CylinderGeometry(S * 0.92, S * 0.92, 1, 6);
  geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.78, metalness: 0 });
  const mesh = new THREE.InstancedMesh(geo, mat, N);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  scene.add(mesh);
  const base = cells.map((c) => new THREE.Color(PAL[c.t] || PAL[0]));
  cells.forEach((c, i) => mesh.setColorAt(i, base[i]));

  // ------------------------------------------------------------------ переходы
  const cur = cells.map((c) => ({ x: c.x, z: c.z, h: reduce ? c.h : 0.01, lift: 0 }));
  let tween = null;
  const dummy = new THREE.Object3D();
  function writeMatrices() {
    for (let i = 0; i < N; i++) {
      const s = cur[i];
      dummy.position.set(s.x, s.lift, s.z);
      dummy.scale.set(1, Math.max(0.01, s.h), 1);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }
  writeMatrices();
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  function startTween(to, { dur = 1300, delay = () => 0, arc = 0, ease = easeInOut, done } = {}) {
    if (reduce) {
      to.forEach((t, i) => Object.assign(cur[i], t, { lift: 0 }));
      writeMatrices();
      mesh.computeBoundingSphere();
      if (done) done();
      return;
    }
    tween = { t0: performance.now(), dur, from: cur.map((s) => ({ ...s })), to, delay: cells.map((c, i) => delay(c, i)), arc, ease, done };
  }
  function stepTween(now) {
    if (!tween) return;
    let active = false;
    for (let i = 0; i < N; i++) {
      const k = Math.min(1, Math.max(0, (now - tween.t0 - tween.delay[i]) / tween.dur));
      if (k < 1) active = true;
      const e = tween.ease(k), a = tween.from[i], b = tween.to[i];
      cur[i].x = a.x + (b.x - a.x) * e;
      cur[i].z = a.z + (b.z - a.z) * e;
      cur[i].h = a.h + (b.h - a.h) * e;
      cur[i].lift = tween.arc * Math.sin(Math.PI * k);
    }
    writeMatrices();
    if (!active) {
      const d = tween.done;
      tween = null;
      mesh.computeBoundingSphere();
      if (d) d();
    }
  }

  // ------------------------------------------------------------------ камера и управление
  const controls = new OrbitControls(camera, canvas);
  // по кругу без ограничений азимута; наклон — от вида сверху почти до горизонта; колесо — к курсору;
  // правая кнопка — сдвиг по земле; клик выбирает МО, только если мышь сдвинулась меньше 5 px
  controls.enableZoom = true;
  controls.zoomToCursor = true;
  controls.minDistance = 90;
  controls.maxDistance = 5600;
  controls.enablePan = true;
  controls.screenSpacePanning = false;
  controls.enableDamping = !reduce;
  controls.dampingFactor = 0.08;
  controls.minPolarAngle = 0.02;
  controls.maxPolarAngle = 1.5;
  if (coarse) {
    // телефон: вертикальный жест листает страницу, горизонтальный поворачивает карту, двумя пальцами — масштаб
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    canvas.style.touchAction = "pan-y";
  }

  const VIEWS = {
    map: { target: new THREE.Vector3(-255, 0, 40), dir: new THREE.Vector3(0, 0.7, 0.71), dist: 1900 },
    islands: { target: new THREE.Vector3(-440, 0, 60), dir: new THREE.Vector3(0, 0.48, 0.88), dist: 3050 },
  };
  let camTween = null;
  const viewPos = (v) => v.target.clone().add(v.dir.clone().normalize().multiplyScalar(v.dist));
  function flyTo(target, pos, dur = 1200) {
    if (reduce) { controls.target.copy(target); camera.position.copy(pos); controls.update(); return; }
    camTween = { t0: performance.now(), dur, ft: controls.target.clone(), fp: camera.position.clone(), tt: target.clone(), tp: pos.clone() };
  }
  function setView(name, dur) { const v = VIEWS[name]; flyTo(v.target, viewPos(v), dur); }
  let narrow = false;
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    narrow = w < 900;
    // на узком экране карта в центре; расстояние — чтобы по ширине поместилась вся карта (половина ширины
    // карты с запасом — 560 единиц, ряд островов — 700) при горизонтальном угле обзора этой камеры
    const tanH = Math.tan((camera.fov * Math.PI) / 360) * camera.aspect;
    const phone = w < 700;
    VIEWS.map.target.x = phone ? -210 : narrow ? 0 : -230;
    VIEWS.map.dir.set(0, phone ? 0.86 : 0.7, phone ? 0.51 : 0.71);  // на телефоне — круче сверху: карта выше в кадре
    // телефон (порция 5b): камера ближе — по ширине кадра около половины карты, европейская часть в центре; остальное — жестом
    VIEWS.map.dist = phone ? Math.max(1100, 290 / tanH) : narrow ? Math.max(2300, 560 / tanH) : 2080;
    VIEWS.islands.target.x = narrow ? 0 : -390;
    VIEWS.islands.dist = narrow ? Math.max(3600, 720 / tanH) : 2750;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(() => {
    resize();
    // пока карту не трогали, вид подстраивается под новый размер (поворот телефона, ширина окна)
    if (!userMoved && selected < 0 && !camTween) { controls.target.copy(VIEWS[mode].target); camera.position.copy(viewPos(VIEWS[mode])); controls.update(); }
  }).observe(stage);
  controls.target.copy(VIEWS.map.target);
  camera.position.copy(viewPos(VIEWS.map));
  controls.update();

  function dolly(f) {
    const d = camera.position.clone().sub(controls.target).multiplyScalar(f);
    flyTo(controls.target.clone(), controls.target.clone().add(d), 450);
  }
  $("#h0-zin").addEventListener("click", () => dolly(0.8));
  $("#h0-zout").addEventListener("click", () => dolly(1.25));
  $("#h0-zreset").addEventListener("click", () => setView(mode, 900));

  // ------------------------------------------------------------------ карта и острова
  let mode = "map";
  const btnTypes = $("#h0-types"), btnMap = $("#h0-map"), labels = $("#h0-labels");
  labels.innerHTML = islands.map((s) =>
    `<div class="isl"><i class="hx${s.t ? "" : " hx0"}" style="--c:${s.color || PAL[0]}"></i><b>${esc(nbsp(s.name))}</b>` +
    `<span class="n">${nf.format(s.n)}</span><span class="note">${esc(nbsp(s.note))}</span></div>`).join("");
  const labelEls = [...labels.children];
  function toIslands(focus) {
    if (mode === "islands") return;
    mode = "islands";
    clearSelection();
    btnTypes.hidden = true; btnMap.hidden = false;
    doc.documentElement.classList.add("h0-islands");
    if (focus) btnMap.focus({ preventScroll: true });
    startTween(cells.map((c) => ({ x: c.ix, z: c.iz, h: c.h })), {
      dur: 1500, arc: 26, delay: (c) => (c.x + W2) * 0.55 + Math.random() * 220,
      done: () => labels.classList.add("show"),
    });
    setView("islands", 1900);
  }
  function toMap(focus, fly = true) {
    if (mode === "map") return;
    mode = "map";
    labels.classList.remove("show");
    btnTypes.hidden = false; btnMap.hidden = true;
    doc.documentElement.classList.remove("h0-islands");
    if (focus) btnTypes.focus({ preventScroll: true });
    startTween(cells.map((c) => ({ x: c.x, z: c.z, h: c.h })), { dur: 1500, arc: 26, delay: () => Math.random() * 500 });
    if (fly) setView("map", 1700);
  }
  btnTypes.addEventListener("click", () => toIslands(true));
  btnMap.addEventListener("click", () => toMap(true));

  // плоская карта (порция 6a): та же SVG-карта ячеек, что без WebGL (landing.js: выбор, эго-сеть, масштаб)
  let flat = false;
  const btnFlat = $("#h0-flat");
  function setFlat(on, focus) {
    if (on && mode === "islands") toMap(false, false);
    flat = on;
    doc.documentElement.classList.toggle("h0-flat", on);
    btnFlat.textContent = on ? btnFlat.dataset.solid : btnFlat.dataset.flat;
    btnTypes.hidden = on || mode === "islands";
    tip.hidden = true;
    if (focus) btnFlat.focus({ preventScroll: true });
  }
  btnFlat.addEventListener("click", () => setFlat(!flat, true));

  // ------------------------------------------------------------------ наведение, выбор
  const ray = new THREE.Raycaster();
  const ptr = new THREE.Vector2();
  const tip = $("#h0-tip");
  let hover = -1, selected = -1;
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const hit = ray.intersectObject(mesh)[0];
    return hit ? hit.instanceId : -1;
  }
  function paint(i, on) {
    if (i < 0 || selected >= 0) return;
    const c = base[i].clone();
    if (on) c.lerp(new THREE.Color("#ffffff"), 0.35);
    mesh.setColorAt(i, c);
    mesh.instanceColor.needsUpdate = true;
  }
  // флаг устойчивости типа (порция 6b, этап usefulness) — строкой после типа; слова — story.card.flag_words
  const FLAG_WORDS = (story.card || {}).flag_words || {};
  const flagText = (f) => (f && FLAG_WORDS[f] ? `<span class="fl">${esc(FLAG_WORDS[f])}</span>` : "");
  function showTip(i, ev) {
    const c = cells[i], r = stage.getBoundingClientRect();
    tip.innerHTML = nbsp(`<b>${esc(cap1(c.name))}</b><span>${esc(c.region)}</span><span class="ty"><i class="hx${c.t ? "" : " hx0"}" style="--c:${PAL[c.t] || PAL[0]}"></i>${esc(typeName(c.t))}</span>${flagText(c.fl)}<span>${esc(ratioText(c.ratio))}</span>`);
    tip.hidden = false;
    const x = Math.min(ev.clientX - r.left + 16, r.width - tip.offsetWidth - 8), y = Math.min(ev.clientY - r.top + 16, r.height - tip.offsetHeight - 8);
    tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }
  canvas.addEventListener("pointermove", (ev) => {
    if (tween || ev.pointerType !== "mouse" || ev.buttons) return;
    const id = pick(ev);
    if (id !== hover) { paint(hover, false); hover = id; paint(hover, true); }
    canvas.style.cursor = id >= 0 ? "pointer" : "grab";
    if (id >= 0) showTip(id, ev); else tip.hidden = true;
  });
  canvas.addEventListener("pointerleave", () => { paint(hover, false); hover = -1; tip.hidden = true; });
  let downAt = null;
  canvas.addEventListener("pointerdown", (ev) => { downAt = [ev.clientX, ev.clientY]; userMoved = true; });
  canvas.addEventListener("click", (ev) => {
    const moved = downAt ? Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) : 0;
    if (moved >= 5 || tween) return;
    const i = pick(ev);
    if (i < 0) return;
    if (ev.pointerType !== "mouse") showTip(i, ev);
    // карточка — та же, что у поиска (landing.js): она сообщит о выборе событием munnet:select
    if (window.munnet && window.munnet.go) window.munnet.go(cells[i].id); else select(i);
  });

  // выбранная ячейка (порция 6a): яркое кольцо с белой подложкой поверх всего и подпись-флажок над столбиком
  const ringMat = (color) => new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
  const ring = new THREE.Group();
  const ringCase = new THREE.Mesh(new THREE.TorusGeometry(S * 1.45, 2.6, 8, 48), ringMat(0xffffff));
  const ringAcc = new THREE.Mesh(new THREE.TorusGeometry(S * 1.45, 1.5, 8, 48), ringMat(0xa3172d));
  ringCase.renderOrder = 10; ringAcc.renderOrder = 11;
  ring.add(ringCase, ringAcc);
  ring.rotation.x = Math.PI / 2;
  ring.visible = false;
  scene.add(ring);
  const flag = $("#h0-flag"), pickedNote = $("#h0-picked-note");
  // выбор: всё, кроме выбранной ячейки, её соседей по своему региону и похожих по тратам, понижено (×LOW) и бледнее —
  // иначе высокие соседи заслоняют выбранную; это показ, а не данные: подпись h0-picked-note говорит об этом,
  // и высоты возвращаются, когда выбор снят
  const LOW = 0.22;
  function spotlight(region, keep = null) {
    const pale = new THREE.Color("#eef0f3");
    cells.forEach((c, j) => {
      const col = base[j].clone();
      if (keep && !keep.has(j)) col.lerp(pale, c.region === region ? 0.55 : 0.85);
      mesh.setColorAt(j, col);
    });
    mesh.instanceColor.needsUpdate = true;
  }
  function heights(keep = null) {
    return cells.map((c, j) => ({ x: c.x, z: c.z, h: keep && !keep.has(j) ? c.h * LOW : c.h }));
  }
  // дуги над картой (сходство трат, не поездки и не потоки; порция 5b): к соседям по своему региону (набор B T7,
  // с ними сверять изменения) — главные, тёмные; к похожим по тратам в других регионах — тонкие и бледные
  const egoMat = new THREE.MeshBasicMaterial({ color: 0x1d1d1d });
  const farMat = new THREE.MeshBasicMaterial({ color: 0x8d96a1, transparent: true, opacity: 0.75 });
  const ego = new THREE.Group();
  scene.add(ego);
  let egoIdx = [];
  function clearEgo() {
    for (const m of ego.children) m.geometry.dispose();
    ego.clear();
    egoIdx = [];
  }
  function arc(a, b, r, m) {
    const d = Math.hypot(b.x - a.x, b.z - a.z);
    const p0 = new THREE.Vector3(a.x, a.h + 2, a.z), p2 = new THREE.Vector3(b.x, b.h + 2, b.z);
    const p1 = new THREE.Vector3((a.x + b.x) / 2, Math.max(a.h, b.h) + 24 + d * 0.35, (a.z + b.z) / 2);
    ego.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(p0, p1, p2), 48, r, 6, false), m));
  }
  function drawEgo(i, simIdx, simbIdx = []) {
    clearEgo();
    egoIdx = [...simbIdx, ...simIdx];
    const a = cells[i]; // положения на карте (выбор всегда возвращает карту из островов)
    for (const j of simIdx) arc(a, cells[j], 0.6, farMat);
    for (const j of simbIdx) arc(a, cells[j], 1.3, egoMat);
  }
  function select(i, simIds = [], simbIds = []) {
    if (mode === "islands") toMap(false, false);
    selected = i;
    hover = -1;
    const c = cells[i];
    ring.visible = true;
    doc.documentElement.classList.add("h0-picked"); // карточка открыта слева: текст экрана под ней прячется
    const simIdx = simIds.map((id) => byId.get(id)).filter((j) => j !== undefined);
    const simbIdx = simbIds.map((id) => byId.get(id)).filter((j) => j !== undefined);
    const keep = new Set([i, ...simIdx, ...simbIdx]);
    spotlight(c.region, keep);
    startTween(heights(keep), { dur: 700, ease: easeOut });
    drawEgo(i, simIdx, simbIdx);
    flag.innerHTML = `<b>${esc(cap1(c.name))}</b><span>${esc(c.region)}</span>`;
    flag.hidden = false;
    pickedNote.hidden = false;
    // кадр (порция 6a): камера выше и дальше, выбранная ячейка — в нижней трети кадра, соседи по своему региону
    // (с ними сверять) — в кадре; на телефоне — почти сверху. e — наибольшее удаление соседа от выбранной ячейки.
    // Видимая глубина земли у цели ≈ 2·dist·tanV / sin φ; выбранная — на −0,3 высоты кадра (нижняя треть, над
    // легендой), соседи помещаются, если половина глубины ≥ e / 0,6; по ширине — с учётом карточки слева.
    const near = (simbIdx.length ? simbIdx : simIdx).map((j) => cells[j]);
    const e = Math.max(60, ...near.map((p) => Math.max(Math.abs(p.x - c.x), Math.abs(p.z - c.z)))) + 30;
    const phone = stage.clientWidth < 700;
    const dir = phone ? new THREE.Vector3(0, 0.985, 0.17) : new THREE.Vector3(0, 0.92, 0.39);
    dir.normalize();
    const sinF = dir.y;
    const card = doc.getElementById("card");
    const cardFrac = !narrow && card && !card.hidden ? Math.min(0.45, card.offsetWidth / Math.max(1, stage.clientWidth)) : 0;
    const tanV = Math.tan((camera.fov * Math.PI) / 360), tanH = tanV * camera.aspect;
    const dist = Math.min(3600, Math.max(phone ? 900 : 950, (e * sinF) / (0.6 * tanV), (e * 1.15) / (tanH * (1 - cardFrac))));
    const hd = (dist * tanV) / sinF;
    const tgt = new THREE.Vector3(c.x - cardFrac * dist * tanH, 0, c.z - 0.3 * hd);
    flyTo(tgt, tgt.clone().add(dir.multiplyScalar(dist)), 1300);
  }
  function clearSelection() {
    if (selected < 0) return;
    selected = -1;
    ring.visible = false;
    flag.hidden = true;
    pickedNote.hidden = true;
    clearEgo();
    doc.documentElement.classList.remove("h0-picked");
    spotlight(null);
    if (mode === "map") startTween(heights(), { dur: 700, ease: easeOut });
  }
  onSelect = (id, sim = [], simb = []) => {
    const i = id == null ? undefined : byId.get(id);
    if (i === undefined) clearSelection(); else select(i, sim, simb);
  };

  // ------------------------------------------------------------------ кадр
  const proj = new THREE.Vector3();
  const cityBox = $("#h0-cities");
  cityBox.innerHTML = CITIES.map((c) => `<span>${esc(c.name)}</span>`).join("");
  const cityEls = [...cityBox.children];
  let cityW = null;
  function placeCities() {
    const show = mode === "map" && selected < 0 && !tween;
    if (cityBox.classList.contains("show") !== show) {
      cityBox.classList.toggle("show", show);
      if (!show) cityEls.forEach((el) => { el.style.visibility = "hidden"; });
    }
    if (!show) return;
    if (!cityW) cityW = cityEls.map((el) => [el.offsetWidth, el.offsetHeight]);
    const w = stage.clientWidth, h = stage.clientHeight, taken = [];
    CITIES.forEach((c, j) => {  // по убыванию населения: крупный город важнее
      const s = cur[c.i];
      proj.set(s.x, s.h + 2, s.z).project(camera);
      const x = (proj.x * 0.5 + 0.5) * w, y = (-proj.y * 0.5 + 0.5) * h;
      const [bw, bh] = cityW[j];
      const box = [x - bw / 2, y - bh - 6, x + bw / 2, y - 6];
      const off = proj.z > 1 || box[0] < 0 || box[2] > w || box[1] < 0 || box[3] > h;
      const hit = taken.some((b) => box[0] < b[2] + 4 && box[2] + 4 > b[0] && box[1] < b[3] + 2 && box[3] + 2 > b[1]);
      const ok = !off && !hit;
      if (ok) taken.push(box);
      cityEls[j].style.visibility = ok ? "visible" : "hidden";
      cityEls[j].style.transform = `translate(${box[0].toFixed(1)}px, ${box[1].toFixed(1)}px)`;
    });
  }
  let userMoved = false, visible = true;
  controls.addEventListener("start", () => { userMoved = true; tip.hidden = true; });
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(stage);
  function frame(now) {
    requestAnimationFrame(frame);
    if ((!visible || flat) && !tween && !camTween) return;
    stepTween(now);
    if (camTween) {
      const k = Math.min(1, (now - camTween.t0) / camTween.dur), e = easeInOut(k);
      controls.target.lerpVectors(camTween.ft, camTween.tt, e);
      camera.position.lerpVectors(camTween.fp, camTween.tp, e);
      if (k >= 1) camTween = null;
    } else if (!reduce && !userMoved && mode === "map" && selected < 0) {
      // лёгкое покачивание, пока карту никто не трогал
      const a = Math.sin(now / 5200) * 0.045, v = VIEWS.map;
      camera.position.copy(v.target).add(viewPos(v).sub(v.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), a));
      controls.target.copy(v.target);
    }
    controls.update();
    if (selected >= 0) {
      const s = cur[selected];
      ring.position.set(s.x, s.h + 1.6, s.z);
      ring.scale.setScalar(reduce ? 1 : 1 + 0.1 * Math.sin(now / 260));
      proj.set(s.x, s.h + 4, s.z).project(camera);
      const off = proj.z > 1 || Math.abs(proj.x) > 1.05 || Math.abs(proj.y) > 1.05;
      flag.style.visibility = off ? "hidden" : "visible";
      flag.style.transform = `translate(${(proj.x * 0.5 + 0.5) * stage.clientWidth}px, ${(-proj.y * 0.5 + 0.5) * stage.clientHeight}px) translate(-50%, calc(-100% - 14px))`;
    }
    if (labels.classList.contains("show")) {
      const w = stage.clientWidth, h = stage.clientHeight;
      islands.forEach((s, j) => {
        proj.set(s.cx, 0, s.cz + s.rad * 0.92).project(camera);
        const dx = j === islands.length - 1 ? -85 : -50, dy = 10 + (j % 2) * (narrow ? 70 : 118);
        labelEls[j].style.transform = `translate(${(proj.x * 0.5 + 0.5) * w}px, ${(-proj.y * 0.5 + 0.5) * h}px) translate(${dx}%, ${dy}px)`;
      });
    }
    placeCities();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);

  // вступление: столбики вырастают волной с запада на восток
  startTween(cells.map((c) => ({ x: c.x, z: c.z, h: c.h })), {
    dur: 1100, ease: easeOut, delay: (c) => (c.x + W2) * 0.9 + Math.random() * 140,
  });

  // адрес: ?mode=islands — сразу разложить по типам; ?mo=<номер или название> — открыть муниципалитет
  const norm = (s) => String(s || "").toLowerCase().replace(/ё/g, "е").trim();
  function moFromParam(v) {
    if (!v) return null;
    if (/^\d+$/.test(v) && byId.has(Number(v))) return Number(v);
    const q = norm(v);
    const c = cells.find((x) => norm(x.name) === q) || cells.find((x) => norm(x.name).startsWith(q));
    return c ? c.id : null;
  }
  setTimeout(() => {
    if (params.get("mode") === "islands") toIslands(false);
    // ?mo= разбирает landing.js (window.munnet.openParam: как поиск; несколько совпадений — список поиска);
    // свой разбор — только если landing.js нет
    const id = window.munnet && window.munnet.openParam ? null : moFromParam(params.get("mo"));
    if (id != null) { if (window.munnet && window.munnet.go) window.munnet.go(id); else select(byId.get(id)); }
    else if (pending != null) onSelect(...pending);
  }, reduce ? 0 : 2300);
  if (pending != null && reduce) onSelect(...pending);
}

main().catch((e) => {
  // объёмная карта не собралась: остаётся статичная (landing.js), страница работает
  console.warn("Корзина и регион: объёмная карта недоступна —", e && e.message ? e.message : e);
  doc.documentElement.classList.remove("has-3d");
  const c = $("#scene"); if (c) c.hidden = true;
  for (const id of ["h0-ui", "h0-actions"]) { const el = doc.getElementById(id); if (el) el.hidden = true; }
});
