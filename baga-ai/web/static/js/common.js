// Общие функции страниц: API, сессия, навигация, коридор цены, карточка объявления.

export const LOGO = `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M2 25 11 11l5 7 4-5 10 12Z" fill="#18202B"/>
<path d="m11 11 2.6 4-2.6 1.6L8.6 15Z" fill="#EEF1F4"/><circle cx="24" cy="8" r="3" fill="#C73E2E"/></svg>`;

export async function api(path, { method = "GET", body, form } = {}) {
  const opt = { method, headers: {}, credentials: "same-origin" };
  if (form) opt.body = form;
  else if (body !== undefined) { opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, opt);
  const isJson = (r.headers.get("content-type") || "").includes("json");
  const data = isJson ? await r.json() : await r.blob();
  if (!r.ok) {
    const err = new Error((isJson && (data.error || data.detail)) || `Ошибка ${r.status}`);
    err.status = r.status;
    throw err;
  }
  return data;
}

let _me;
export async function me(force = false) {
  if (!_me || force) _me = await api("/api/me").catch(() => ({ role: "guest", permissions: [] }));
  return _me;
}
export const can = (m, p) => (m.permissions || []).includes(p);

export function nav(current) {
  const el = document.querySelector("header.nav");
  if (!el) return;
  el.innerHTML = `<div class="wrap">
    <a class="brand" href="/">${LOGO}<span>Baga</span></a>
    <nav class="nav-links" aria-label="Разделы">
      <a href="/#how" class="hide-m">Как это работает</a>
      <a href="/app" ${current === "app" ? 'aria-current="page"' : ""}>Поиск</a>
      <a href="/app#price" class="hide-m">Проверить цену</a>
      <span id="nav-user"></span>
    </nav></div>`;
  const onScroll = () => el.classList.toggle("scrolled", scrollY > 8);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  me().then((m) => {
    const slot = el.querySelector("#nav-user");
    if (m.user) {
      slot.innerHTML = `${m.role === "admin" ? `<a href="/admin" ${current === "admin" ? 'aria-current="page"' : ""}>Панель</a> ` : ""}
        <button class="btn quiet small" id="logout" title="Вы вошли как ${esc(m.user)}">Выйти</button>`;
      slot.style.display = "flex"; slot.style.gap = "18px"; slot.style.alignItems = "center";
      slot.querySelector("#logout").onclick = async () => { await api("/api/auth/logout", { method: "POST" }); location.reload(); };
    } else {
      slot.innerHTML = `<a class="btn small" href="/login?next=${encodeURIComponent(location.pathname + location.hash)}">Войти</a>`;
    }
  });
}

export function toast(text, ms = 2600) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); document.body.append(t); }
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), ms);
}

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const money = (x) => (x == null ? "—" : Math.round(x).toLocaleString("ru-RU").replace(/,/g, " ") + " ₸");
export const thousands = (x) => (x == null ? "—" : `${Math.round(x / 1000).toLocaleString("ru-RU")} тыс.`);

export function verdictClass(price, band) {
  if (!band || price == null) return "within";
  if (price > band.p90) return "above";
  if (price < band.p10) return "below";
  return "within";
}

export function verdictText(price, band) {
  const cls = verdictClass(price, band);
  if (!band || price == null) return "";
  if (cls === "above") return `дороже похожих на ${Math.round((price / band.p90 - 1) * 100)}%`;
  if (cls === "below") return `дешевле похожих на ${Math.round((1 - price / band.p10) * 100)}%`;
  return "в коридоре похожих";
}

/** Коридор цены. Шкала — от 0.6·p10 до 1.4·p90, чтобы точка «сильно дороже» не упиралась в край. */
export function bandHTML(price, band, { legend = true, animate = false } = {}) {
  if (!band || band.p10 == null) return `<p class="small muted">Похожих объявлений мало — диапазон не считаем.</p>`;
  const lo = Math.min(band.p10 * 0.6, price ?? Infinity), hi = Math.max(band.p90 * 1.4, price ?? 0);
  const pos = (x) => (100 * (x - lo)) / (hi - lo);
  const cls = verdictClass(price, band);
  const dot = price == null ? "" :
    `<span class="dot ${cls}" style="left:${animate ? pos(band.p50) : pos(price)}%" data-to="${pos(price)}"></span>`;
  return `<div class="band" role="img" aria-label="Похожие сдают за ${money(band.p10)} – ${money(band.p90)}${price ? `, это объявление — ${money(price)}` : ""}">
      <span class="track"></span>
      <span class="range" style="left:${pos(band.p10)}%;width:${pos(band.p90) - pos(band.p10)}%"></span>
      <span class="median" style="left:${pos(band.p50)}%"></span>${dot}
    </div>${legend ? `<div class="band-legend"><span>похожие: ${thousands(band.p10)} – ${thousands(band.p90)}</span>
      <span class="verdict ${cls}">${verdictText(price, band)}</span></div>` : ""}`;
}

export function releaseDots(root = document) {
  requestAnimationFrame(() => requestAnimationFrame(() =>
    root.querySelectorAll(".band .dot[data-to]").forEach((d) => { d.style.left = d.dataset.to + "%"; })));
}

export function cardHTML(c, { explain = true } = {}) {
  const meta = [c.rooms && `${c.rooms}-комн.`, c.area && `${Math.round(c.area)} м²`,
    c.floor && c.floors_total && `${Math.round(c.floor)}/${Math.round(c.floors_total)} эт.`,
    c.district && c.district !== "не указан" ? c.district.replace(" р-н", " район") : c.city].filter(Boolean).join(", ");
  const tag = c.matched?.length ? `совпало фото: ${c.matched.slice(0, 2).join(", ")}` : c.by_description ? "совпало описание" : "";
  return `<article class="card" id="l-${esc(c.id)}" data-id="${esc(c.id)}">
    <div class="ph">${c.photo ? `<img src="${esc(c.photo)}" alt="Фото квартиры ${esc(meta)}" loading="lazy" referrerpolicy="no-referrer">` : ""}
      ${tag ? `<span class="tag">${esc(tag)}${c.coverage ? ` · ${esc(c.coverage)}` : ""}</span>` : ""}</div>
    <div class="price">${money(c.price)}<small> в месяц</small></div>
    <div class="meta">${esc(meta)}${c.duplicates > 1 ? `<br><span class="muted">то же объявление встречается ${c.duplicates} раз</span>` : ""}</div>
    ${bandHTML(c.price, c.band)}
    <div class="actions">
      ${c.url ? `<a class="btn quiet" href="${esc(c.url)}" target="_blank" rel="noopener">Открыть на krisha.kz</a>` : ""}
      ${explain ? `<button class="btn quiet" data-why="${esc(c.id)}" aria-expanded="false">Почему такая цена</button>` : ""}
    </div>
    <div class="why" hidden></div>
  </article>`;
}

/** «Почему такая цена» — раскрывается в карточке, повторный клик сворачивает. */
export function wireExplain(root) {
  root.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-why]");
    if (!b) return;
    const box = b.closest(".card").querySelector(".why");
    if (!box.hidden) { box.hidden = true; b.setAttribute("aria-expanded", "false"); return; }
    box.hidden = false; b.setAttribute("aria-expanded", "true");
    if (box.dataset.loaded) return;
    box.textContent = "Сравниваю с похожими…";
    try {
      const r = await api(`/api/listing/${b.dataset.why}/explain`);
      box.innerHTML = linkify(esc(r.text)) + (r.llm ? "" : `<br><a class="small" href="/login">Войдите</a><span class="small muted">, чтобы получать подробный разбор.</span>`);
      box.dataset.loaded = "1";
    } catch (err) { box.textContent = err.message; }
  });
}

/** [1015807119] в тексте ответа -> ссылка на карточку ниже. */
export function linkify(html, { external = false } = {}) {
  return html.replace(/\[(\d{6,})\]/g, (_, id) => external
    ? `<a href="https://krisha.kz/a/show/${id}" target="_blank" rel="noopener">[${id}]</a>`
    : `<a href="#l-${id}" data-jump="${id}">[${id}]</a>`);
}

export function wireJumps(root) {
  root.addEventListener("click", (e) => {
    const a = e.target.closest("[data-jump]");
    if (!a) return;
    const card = document.getElementById("l-" + a.dataset.jump);
    if (!card) return;
    e.preventDefault();
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1400);
  });
}
