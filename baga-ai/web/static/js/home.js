import { api, bandHTML, esc, money, nav, releaseDots, verdictClass, verdictText } from "./common.js";

nav("home");

const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, reduce ? 0 : ms));

// ---------------------------------------------------------------- демо в hero
// Одна срежиссированная сцена: запрос печатается, пока идёт настоящий поиск по базе,
// потом квартиры появляются по одной, и точка цены каждой въезжает в её коридор.
const demoEl = document.querySelector(".demo");
const typed = demoEl.querySelector(".demo-typed");
const grid = demoEl.querySelector(".demo-grid");
const status = demoEl.querySelector(".demo-status");
const replay = demoEl.querySelector(".demo-replay");
const DEMO_QUERY = "двушка в Алматы до 450 тысяч, светлая кухня, белые фасады";
let demoData = null;

async function type(text) {
  typed.textContent = "";
  if (reduce) { typed.textContent = text; return; }
  for (const ch of text) {
    typed.textContent += ch;
    await sleep(ch === "," ? 180 : 34 + Math.random() * 30);
  }
}

function tile(c) {
  const cls = verdictClass(c.price, c.band);
  return `<a class="demo-item" href="${esc(c.url || "/app")}" target="_blank" rel="noopener"
      title="${esc(`${money(c.price)} — ${verdictText(c.price, c.band) || "диапазон не посчитан"}`)}">
    <div class="ph">${c.photo ? `<img src="${esc(c.photo)}" alt="Кухня в найденной квартире" referrerpolicy="no-referrer">` : ""}</div>
    <span class="p">${money(c.price)}</span>
    ${bandHTML(c.price, c.band, { legend: false, animate: true })}
    <span class="sr">${esc(verdictText(c.price, c.band))}</span>
  </a>`.replace('class="demo-item"', `class="demo-item v-${cls}"`);
}

async function playDemo() {
  replay.hidden = true;
  status.textContent = "Ищем по фотографиям кухонь…";
  grid.innerHTML = '<div class="demo-tile skeleton"></div>'.repeat(6);
  const dataP = demoData ? Promise.resolve(demoData) : api("/api/demo");
  await type(DEMO_QUERY);
  try {
    demoData = await dataP;
  } catch {
    status.textContent = "Демо сейчас не загрузилось — попробуйте поиск сами.";
    return;
  }
  const items = demoData.results.slice(0, 6);
  // картинки грузим заранее, чтобы появление не мигало пустыми рамками
  await Promise.all(items.map((c) => new Promise((ok) => {
    if (!c.photo) return ok();
    const im = new Image(); im.referrerPolicy = "no-referrer"; im.onload = im.onerror = ok; im.src = c.photo;
    setTimeout(ok, 2500);
  })));
  grid.innerHTML = items.map(tile).join("");
  const els = [...grid.children];
  for (const el of els) { el.classList.add("in"); await sleep(140); }
  await sleep(250);
  releaseDots(grid);
  const n = items.filter((c) => c.band).length;
  const above = items.filter((c) => verdictClass(c.price, c.band) === "above").length;
  status.textContent = above
    ? `Нашлось ${items.length} квартир, ${above} из них дороже похожих.`
    : `Нашлось ${items.length} квартир, все не дороже похожих.`;
  if (n) replay.hidden = false;
}

replay.addEventListener("click", playDemo);
// запускаем, когда демо видно (на телефоне оно ниже первого экрана)
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); playDemo(); }
}, { threshold: 0.25 }).observe(demoEl);

// ---------------------------------------------------------------- разбор запроса
// Наведение (или фокус) на чип подсвечивает кусок запроса, из которого он получился.
const parse = document.querySelector(".parse-demo");
function light(k) {
  parse.querySelectorAll("[data-k]").forEach((el) => el.classList.toggle("on", el.dataset.k === k));
}
parse.querySelectorAll("[data-k]").forEach((el) => {
  el.addEventListener("mouseenter", () => light(el.dataset.k));
  el.addEventListener("focus", () => light(el.dataset.k));
  el.addEventListener("mouseleave", () => light(null));
  el.addEventListener("blur", () => light(null));
});

// ---------------------------------------------------------------- коридор цены
const BAND = { p10: 318000, p50: 385000, p90: 468000 };
const range = document.getElementById("price-range");
const out = document.getElementById("price-out");
const slot = document.getElementById("price-band-slot");
function drawBand() {
  const price = Number(range.value);
  out.textContent = money(price);
  slot.innerHTML = bandHTML(price, BAND);
  range.setAttribute("aria-valuetext", `${money(price)}: ${verdictText(price, BAND)}`);
}
range.addEventListener("input", drawBand);
drawBand();
