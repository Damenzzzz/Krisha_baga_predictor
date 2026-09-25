import { api, bandHTML, can, cardHTML, esc, linkify, me, money, nav, releaseDots, thousands, toast,
  verdictClass, verdictText, wireExplain, wireJumps } from "./common.js";

nav("app");

const $ = (s, r = document) => r.querySelector(s);
const results = $("#results");
wireExplain(results);
wireJumps(document);

const ROOMS_PHOTO = { kitchen: "кухня", living_room: "гостиная", bedroom: "спальня", bathroom: "санузел", hallway: "прихожая", balcony: "балкон" };
const MODES = ["assistant", "describe", "photo", "price"];
let viewer = { role: "guest", permissions: [] };
let meta = null;

// ---------------------------------------------------------------- режимы

function setMode(mode, { push = true } = {}) {
  if (!MODES.includes(mode)) mode = can(viewer, "assistant") ? "assistant" : "describe";
  for (const m of MODES) {
    const tab = $(`#t-${m}`), panel = $(`#p-${m}`);
    tab.setAttribute("aria-selected", String(m === mode));
    tab.tabIndex = m === mode ? 0 : -1;
    panel.hidden = m !== mode;
  }
  results.innerHTML = "";
  const out = { assistant: "#assistant-out", describe: "#describe-out", photo: "#photo-out", price: "#price-out" };
  Object.values(out).forEach((s) => ($(s).innerHTML = ""));
  if (push) history.replaceState(null, "", `/app#${mode}`);
  const input = { assistant: "#ask-q", describe: "#desc-q", price: "#price-url" }[mode];
  if (input && !$(`#p-${mode} .locked:not([hidden])`)) $(input).focus({ preventScroll: true });
}

document.querySelector(".modes").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (b) setMode(b.dataset.mode);
});
document.querySelector(".modes").addEventListener("keydown", (e) => {
  if (!["ArrowRight", "ArrowLeft"].includes(e.key)) return;
  const i = MODES.indexOf(document.activeElement.dataset.mode);
  const next = MODES[(i + (e.key === "ArrowRight" ? 1 : MODES.length - 1)) % MODES.length];
  setMode(next);
  $(`#t-${next}`).focus();
});

// ---------------------------------------------------------------- общие куски

const busy = (el, text) => (el.innerHTML = `<div class="status"><span class="spinner" aria-hidden="true"></span>${esc(text)}</div>`);
const fail = (el, err) => (el.innerHTML = `<p class="error">${esc(err.message || err)}</p>`);

function showResults(list, head) {
  results.innerHTML = list.length
    ? (head ? `<p class="results-head">${esc(head)}</p>` : "") + list.map((c) => cardHTML(c)).join("")
    : "";
  releaseDots(results);
}

function feedbackRow(kind, query, answer, traceId) {
  if (!can(viewer, "feedback")) return "";
  return `<div class="row fb" data-kind="${kind}" data-trace="${esc(traceId || "")}">
      <span class="small muted">Ответ помог?</span>
      <button class="thumb-btn" data-v="1" aria-pressed="false" aria-label="Да, помог">👍</button>
      <button class="thumb-btn" data-v="0" aria-pressed="false" aria-label="Нет, не помог">👎</button>
    </div>`.replace("<div", `<div data-q="${esc(query)}" data-a="${esc(answer.slice(0, 1500))}"`);
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest(".fb .thumb-btn");
  if (!b) return;
  const row = b.closest(".fb");
  if (row.dataset.sent) return;
  row.dataset.sent = "1";
  row.querySelectorAll(".thumb-btn").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  try {
    await api("/api/feedback", { method: "POST", body: { kind: row.dataset.kind, value: Number(b.dataset.v),
      trace_id: row.dataset.trace || null, query: row.dataset.q, answer: row.dataset.a } });
    toast("Спасибо, отзыв записан");
  } catch (err) { toast(err.message); delete row.dataset.sent; }
});

// ---------------------------------------------------------------- ассистент

const assistantOut = $("#assistant-out");
let lastAnswer = "";

async function runAssistant(query) {
  if (!query.trim()) return;
  results.innerHTML = "";
  busy(assistantOut, "Разбираю запрос и ищу по фотографиям…");
  try {
    renderAgent(await api("/api/assistant", { method: "POST", body: { query } }), query);
  } catch (err) { fail(assistantOut, err); }
}

function renderAgent(r, query) {
  if (r.status === "paused") {
    assistantOut.innerHTML = `<div class="answer pause">
      <h3>${esc(r.question)}</h3>
      <div class="conds">${r.assumptions.map((a) => `<span class="cond">${esc(a)}</span>`).join("")}</div>
      <div class="row">
        <button class="btn" data-resume="1">Да, искать с ними</button>
        <button class="btn ghost" data-resume="0">Нет, убрать</button>
      </div></div>`;
    assistantOut.querySelectorAll("[data-resume]").forEach((b) => (b.onclick = async () => {
      busy(assistantOut, "Ищу…");
      try {
        renderAgent(await api("/api/assistant/resume", { method: "POST", body: { thread_id: r.thread_id, confirmed: b.dataset.resume === "1" } }), query);
      } catch (err) { fail(assistantOut, err); }
    }));
    assistantOut.querySelector("[data-resume]").focus();
    return;
  }
  lastAnswer = r.answer;
  const f = r.filters || {};
  const conds = [
    f.city && f.city[0].toUpperCase() + f.city.slice(1),
    ...(f.districts || []).map((d) => d[0].toUpperCase() + d.slice(1)),
    f.rooms && `${f.rooms.join(" или ")} комн.`,
    f.price_min && `от ${thousands(f.price_min)}`, f.price_max && `до ${thousands(f.price_max)}`,
    f.area_min && `от ${f.area_min} м²`, f.not_first_floor && "не первый этаж", f.not_last_floor && "не последний этаж",
    // если модель не выделила стиль, в поиск по фото уходит весь запрос — чип его не дублирует
    r.style && r.style.trim().toLowerCase() !== query.trim().toLowerCase() && `на фото: ${r.style}`,
  ].filter(Boolean);
  assistantOut.innerHTML = `<div class="answer">
    ${conds.length ? `<div class="conds" aria-label="Условия поиска">${conds.map((c) => `<span class="cond">${esc(c)}</span>`).join("")}</div>` : ""}
    <div class="text">${linkify(esc(r.answer)).replace(/\n\n/g, "<br><br>")}</div>
    <div class="row">
      ${feedbackRow("assistant", query, r.answer, r.trace_id)}
      ${can(viewer, "voice") ? `<button class="btn quiet" id="speak">Послушать ответ</button>` : ""}
    </div>
    <details class="log"><summary>Как ассистент искал</summary><ol>${(r.log || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ol></details>
  </div>`;
  const sp = $("#speak");
  if (sp) sp.onclick = () => speak(r.answer, sp);
  showResults(r.results);
}

$("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); runAssistant($("#ask-q").value); });

async function speak(text, btn) {
  btn.disabled = true;
  btn.textContent = "Готовлю озвучку…";
  try {
    const blob = await api("/api/voice/speak", { method: "POST", body: { text } });
    const audio = new Audio(URL.createObjectURL(blob));
    btn.textContent = "Звучит";
    audio.onended = () => { btn.textContent = "Послушать ещё раз"; btn.disabled = false; };
    await audio.play();
  } catch (err) { toast(err.message); btn.textContent = "Послушать ответ"; btn.disabled = false; }
}

// голос: запись в браузере -> расшифровка на сервере -> тот же запрос ассистенту
const mic = $("#mic");
let rec = null;
mic.addEventListener("click", async () => {
  if (rec) { rec.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia) { toast("Браузер не даёт доступ к микрофону"); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch { toast("Разрешите доступ к микрофону в настройках браузера"); return; }
  const chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    mic.classList.remove("rec"); mic.setAttribute("aria-label", "Сказать голосом");
    const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
    rec = null;
    busy(assistantOut, "Распознаю речь…");
    const form = new FormData();
    form.append("audio", blob, "voice." + (blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm"));
    try {
      const { text } = await api("/api/voice/transcribe", { method: "POST", form });
      if (!text) { assistantOut.innerHTML = `<p class="error">Не расслышал — попробуйте ещё раз, ближе к микрофону.</p>`; return; }
      $("#ask-q").value = text;
      await runAssistant(text);
      const sp = $("#speak");
      if (sp && lastAnswer) speak(lastAnswer, sp);
    } catch (err) { fail(assistantOut, err); }
  };
  rec.start();
  mic.classList.add("rec");
  mic.setAttribute("aria-label", "Остановить запись");
  toast("Говорите. Нажмите ещё раз, чтобы закончить.");
});

// ---------------------------------------------------------------- по описанию

function fillSelect(sel, values, { keepFirst = true } = {}) {
  const first = keepFirst ? sel.querySelector("option") : null;
  sel.innerHTML = "";
  if (first) sel.append(first);
  for (const v of values) sel.append(new Option(v[0].toUpperCase() + v.slice(1), v));
}

function fillToggles(box, items) {
  box.innerHTML = items.map(([v, label]) =>
    `<label><input type="checkbox" value="${esc(v)}"><span>${esc(label)}</span></label>`).join("");
}

function readFilters(form) {
  const body = {};
  const city = form.querySelector("[name=city]").value;
  if (city) body.city = city;
  const pm = form.querySelector("[name=price_max]").value, am = form.querySelector("[name=area_min]").value;
  if (pm) body.price_max = Number(pm);
  if (am) body.area_min = Number(am);
  for (const box of form.querySelectorAll(".toggles")) {
    const vals = [...box.querySelectorAll("input:checked")].map((i) => i.value);
    if (vals.length) body[box.dataset.name] = box.dataset.name === "rooms" ? vals.map(Number) : vals;
  }
  return body;
}

function summarizeFilters() {
  const b = readFilters($("#desc-filters"));
  const parts = [b.city, b.rooms && `${b.rooms.join(", ")} комн.`, b.price_max && `до ${thousands(b.price_max)}`,
    b.area_min && `от ${b.area_min} м²`, b.districts && `${b.districts.length} р-н`,
    b.photo_rooms && b.photo_rooms.map((r) => ROOMS_PHOTO[r]).join(", ")].filter(Boolean);
  $("#desc-filters .filters-sum").textContent = parts.length ? `— ${parts.join(", ")}` : "— не заданы";
}
$("#desc-filters").addEventListener("change", summarizeFilters);

const describeOut = $("#describe-out");
$("#desc-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#desc-q").value.trim();
  const body = { query: query || null, ...readFilters($("#desc-filters")) };
  busy(describeOut, "Ищу по фотографиям и описаниям…");
  results.innerHTML = "";
  try {
    const r = await api("/api/search", { method: "POST", body });
    describeOut.innerHTML = r.answer || r.warnings?.length ? `<div class="answer">
        ${r.warnings?.length ? `<p class="warn">${esc(r.warnings.join("; "))}</p>` : ""}
        ${r.answer ? `<div class="text">${linkify(esc(r.answer))}</div>${feedbackRow("search", query, r.answer, r.trace_id)}` : ""}
      </div>` : "";
    if (!r.results.length) describeOut.innerHTML += `<div class="empty">По этим условиям ничего нет. Уберите район или поднимите цену.</div>`;
    showResults(r.results, `Подошло по условиям: ${r.n_candidates.toLocaleString("ru-RU")}. Показаны самые похожие.`);
  } catch (err) { fail(describeOut, err); }
});

// ---------------------------------------------------------------- по фото

const files = [];
const thumbs = $("#thumbs"), drop = $("#drop"), photoOut = $("#photo-out");
function drawThumbs() {
  thumbs.innerHTML = files.map((f, i) => `<figure><img src="${URL.createObjectURL(f)}" alt="Ваше фото ${i + 1}">
    <button type="button" data-rm="${i}" aria-label="Убрать фото ${i + 1}">×</button></figure>`).join("");
  $("#photo-go").disabled = !files.length;
}
function addFiles(list) {
  for (const f of list) if (f.type.startsWith("image/") && files.length < 6) files.push(f);
  if (list.length && files.length >= 6) toast("Не больше 6 фото за раз");
  drawThumbs();
}
$("#photo-files").addEventListener("change", (e) => addFiles(e.target.files));
thumbs.addEventListener("click", (e) => { const b = e.target.closest("[data-rm]"); if (b) { files.splice(+b.dataset.rm, 1); drawThumbs(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

$("#photo-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData();
  files.forEach((f) => form.append("images", f, f.name));
  const city = e.target.city.value, pm = e.target.price_max.value;
  if (city) form.append("city", city);
  if (pm) form.append("price_max", pm);
  busy(photoOut, "Определяю комнату на каждом фото и ищу похожие…");
  results.innerHTML = "";
  try {
    const r = await api("/api/search/photo", { method: "POST", form });
    photoOut.innerHTML = r.query_rooms?.length
      ? `<p class="muted">На ваших фото: ${esc(r.query_rooms.join(", "))}. Каждое искали среди таких же комнат.</p>` : "";
    showResults(r.results, `Проверено объявлений: ${r.n_candidates.toLocaleString("ru-RU")}.`);
  } catch (err) { fail(photoOut, err); }
});

// ---------------------------------------------------------------- проверка цены

const priceOut = $("#price-out");
function renderPrice(r) {
  const price = r.listing?.price;
  const cls = verdictClass(price, r.band);
  const title = price
    ? (cls === "within" ? "Цена в коридоре похожих" : verdictText(price, r.band)[0].toUpperCase() + verdictText(price, r.band).slice(1))
    : `Похожие сдают за ${thousands(r.band.p10)} – ${thousands(r.band.p90)}`;
  const l = r.listing || {};
  priceOut.innerHTML = `<div class="verdict-card">
      <div>
        <p class="muted">${esc([l.rooms && `${l.rooms}-комн.`, l.area && `${Math.round(l.area)} м²`, l.district || l.city].filter(Boolean).join(", "))}</p>
        <p class="big verdict ${cls}">${esc(title)}</p>
        ${price ? `<p class="muted">В объявлении: ${money(price)} в месяц</p>` : ""}
        ${bandHTML(price, r.band)}
        ${r.reliable === false ? `<p class="small warn">Похожих объявлений мало, оценка ненадёжна.</p>` : ""}
      </div>
      <div>
        <p>${linkify(esc(r.text), { external: true })}</p>
        ${r.comparables?.length ? `<p class="small muted" style="margin-top:14px">Похожие объявления, на которых построен вывод:</p>
        <ul class="comps">${r.comparables.map((c) => `<li><a href="https://krisha.kz/a/show/${esc(c.id)}" target="_blank" rel="noopener">${c.rooms}-комн., ${Math.round(c.area)} м², ${esc(c.district || "")}</a><span>${money(c.price)}</span></li>`).join("")}</ul>` : ""}
        ${feedbackRow("price", l.listing_id || "параметры", r.text, null)}
      </div>
    </div>`;
  releaseDots(priceOut);
}

$("#price-link").addEventListener("submit", async (e) => {
  e.preventDefault();
  const link = $("#price-url").value.trim();
  if (!link) return;
  busy(priceOut, "Сравниваю с похожими объявлениями…");
  try { renderPrice(await api("/api/price", { method: "POST", body: { link } })); }
  catch (err) { fail(priceOut, err); if (err.status === 404) $("#price-manual").open = true; }
});
$("#price-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = Object.fromEntries([...new FormData(e.target)].filter(([, v]) => v !== "")
    .map(([k, v]) => [k, ["city", "district"].includes(k) ? v : Number(v)]));
  busy(priceOut, "Оцениваю…");
  try { renderPrice(await api("/api/price", { method: "POST", body })); }
  catch (err) { fail(priceOut, err); }
});

// ---------------------------------------------------------------- старт

(async () => {
  [viewer, meta] = await Promise.all([me(), api("/api/meta").catch(() => null)]);
  if (!can(viewer, "assistant")) {
    $("#ask-form").hidden = true;
    $("#p-assistant .hint").hidden = true;
    $("#p-assistant .locked").hidden = false;
  }
  if (meta) {
    document.querySelectorAll("select[name=city]").forEach((s) => fillSelect(s, meta.cities, { keepFirst: s.required !== true }));
    fillSelect($("#price-form [name=district]"), meta.districts);
    fillToggles($("#desc-filters [data-name=rooms]"), meta.rooms.filter((r) => r <= 5).map((r) => [r, r === 5 ? "5+" : r]));
    fillToggles($("#desc-filters [data-name=districts]"), meta.districts.map((d) => [d, d.replace(" р-н", "")]));
    fillToggles($("#desc-filters [data-name=photo_rooms]"), Object.entries(ROOMS_PHOTO));
  }
  summarizeFilters();
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  let mode = location.hash.slice(1);
  if (!MODES.includes(mode)) mode = can(viewer, "assistant") ? "assistant" : "describe";
  setMode(mode, { push: false });
  if (q) {
    history.replaceState(null, "", `/app#${mode}`);
    if (mode === "assistant" && can(viewer, "assistant")) { $("#ask-q").value = q; runAssistant(q); }
    else { setMode("describe"); $("#desc-q").value = q; $("#desc-form").requestSubmit(); }
  }
})();
addEventListener("hashchange", () => { const m = location.hash.slice(1); if (MODES.includes(m)) setMode(m, { push: false }); });
