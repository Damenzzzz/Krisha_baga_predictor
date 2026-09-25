"""LoRA-адаптация текстовой башни SigLIP 2 под язык объявлений krisha.kz.

Гипотеза. SigLIP 2 учили на англоязычных подписях из интернета. Наши запросы — русский
язык аренды: «евроремонт», «хрущёвка», «кухня-гостиная», «с/у раздельный». Если подстроить
текстовую башню на парах «описание объявления -> фотографии этого объявления», русский
запрос будет точнее попадать в нужные фото.

Почему именно так:
  * LoRA только на ТЕКСТОВУЮ башню, визуальная заморожена -> 73 тысячи уже посчитанных
    эмбеддингов фото остаются валидными, пересчитывать галерею не нужно. Адаптер ~2 МБ.
  * Цель для описания — средний эмбеддинг фото объявления (без рендеров и битых фото).
  * Лосс — сигмоидный, как при обучении самой SigLIP, с её же logit_scale и logit_bias.
  * Запросы короткие, описания длинные: на каждом шаге берётся случайный кусок описания
    из 1–2 предложений — модель учится на коротких фразах.
  * Разбиение групповое (splits.py): дубли объявления не попадают в train и test сразу,
    иначе recall завышался бы запоминанием.

Оценка на отложенном фолде (модель эти объявления не видела):
  1. описание -> фото своего объявления: recall@1, recall@10 среди объявлений теста;
  2. golden-запросы из evals.py с автоматической разметкой: precision@5 по фото среди
     объявлений теста — ровно то, что видит пользователь.
  3. визуальный судья (Gemini смотрит на фото) на всех 32 golden-запросах среди объявлений теста.
Адаптер включается (USE_TEXT_LORA=1), только если прирост на визуальном судье больше шума
(+5 п.п. при 160 вердиктах) и автоматическая разметка не хуже.

Запуск: python finetune_text_lora.py [--epochs 3] [--eval-only]
"""
import argparse
import json
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import ART_DIR, CLIP_MODEL_ID, DEVICE, RENDER_THR, SEED

ADAPTER_DIR = ART_DIR / "siglip_text_lora"
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def lora_config(r=16):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=TARGETS, bias="none")


def attach_adapter(model, path=ADAPTER_DIR):
    """Вставляет сохранённый LoRA в text_model уже загруженной SigLIP (для embed.py)."""
    from peft import inject_adapter_in_model, set_peft_model_state_dict
    meta = json.loads((path / "meta.json").read_text())
    inject_adapter_in_model(lora_config(meta["r"]), model.text_model)
    state = torch.load(path / "adapter.pt", map_location="cpu")
    set_peft_model_state_dict(model.text_model, state)
    return model


# --------------------------------------------------------------- данные

def load_data():
    from embed import load_embeddings
    from listings import load_listings
    from rooms import load_rooms
    from splits import add_folds
    idx, E, valid = load_embeddings()
    r = load_rooms()
    ok = valid & (r.render_prob.to_numpy() < RENDER_THR)
    lids = idx.listing_id.to_numpy()
    df = add_folds(load_listings())                 # add_folds сбрасывает индекс — ставим id после
    df = df.set_index(df.listing_id.astype(str), drop=False)
    return idx, E, ok, lids, r.room_type.to_numpy(), df


def listing_targets(E, ok, lids) -> dict[str, np.ndarray]:
    out = {}
    order = np.argsort(lids, kind="stable")
    sl, se, so = lids[order], E[order], ok[order]
    bounds = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = so[a:b]
        if m.any():
            v = se[a:b][m].mean(0)
            out[str(sl[a])] = v / (np.linalg.norm(v) + 1e-12)
    return out


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text or "") if len(s.strip()) >= 12]
    return parts or ([text.strip()] if text and text.strip() else [])


def sample_text(desc: str, rng: random.Random) -> str:
    s = sentences(desc)
    if len(s) <= 1:
        return s[0] if s else ""
    i = rng.randrange(len(s))
    return " ".join(s[i:i + rng.choice((1, 2))])


# --------------------------------------------------------------- модель

def encode(model, proc, texts, grad=False):
    inp = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64,
               truncation=True, return_tensors="pt").to(DEVICE)
    with torch.set_grad_enabled(grad):
        out = model.get_text_features(**inp)
        v = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(v.float(), dim=-1)


def train(model, proc, pairs, targets, epochs=3, batch=96, lr=2e-4, seed=SEED):
    from peft import get_peft_model_state_dict, inject_adapter_in_model
    for p in model.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(lora_config(), model.text_model)
    params = [p for n, p in model.text_model.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad_(True)
    print(f"обучаемых параметров: {sum(p.numel() for p in params):,} "
          f"из {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scale, bias = model.logit_scale.exp().detach(), model.logit_bias.detach()
    rng = random.Random(seed)
    steps = epochs * (len(pairs) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    t0, step = time.time(), 0
    for ep in range(epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs) - batch + 1, batch):
            chunk = pairs[i:i + batch]
            txt = [sample_text(d, rng) for _, d in chunk]
            img = torch.tensor(np.stack([targets[l] for l, _ in chunk]), device=DEVICE)
            t = encode(model, proc, txt, grad=True)
            logits = t @ img.T * scale + bias
            labels = 2 * torch.eye(len(chunk), device=DEVICE) - 1
            loss = -F.logsigmoid(labels * logits).sum() / len(chunk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                print(f"  эпоха {ep + 1} шаг {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f} с)")
    model.eval()
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(model.text_model), ADAPTER_DIR / "adapter.pt")
    return model


# --------------------------------------------------------------- оценка

@torch.no_grad()
def eval_desc_to_photo(model, proc, pairs, targets):
    ids = [l for l, _ in pairs]
    T = np.stack([targets[l] for l in ids])
    Q = torch.cat([encode(model, proc, [d[:400] for _, d in pairs[i:i + 128]]) for i in range(0, len(pairs), 128)])
    S = Q.cpu().numpy() @ T.T
    rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)
    return {"recall@1": round(float((rank < 1).mean()) * 100, 1), "recall@10": round(float((rank < 10).mean()) * 100, 1),
            "n": len(pairs)}


@torch.no_grad()
def eval_golden(model, proc, E, ok, lids, room, test_ids, df, k=5):
    from evals import GOLDEN
    test_mask = np.isin(lids, list(test_ids)) & ok
    desc = df.description.fillna("").str.lower()
    rows = []
    for g in [g for g in GOLDEN if g["pattern"]]:
        q = encode(model, proc, [g["ru"]]).cpu().numpy()[0]
        mask = test_mask & (np.isin(room, g["room"]) if g["room"] else True)
        s = np.where(mask, E @ q, -np.inf)
        found, seen = [], set()
        for i in np.argsort(-s):
            if not np.isfinite(s[i]):
                break
            if lids[i] not in seen:
                seen.add(lids[i])
                found.append(lids[i])
                if len(found) == k:
                    break
        hits = [bool(re.search(g["pattern"], desc.get(l, ""))) for l in found]
        base = float(desc[desc.index.isin(test_ids)].str.contains(g["pattern"], regex=True).mean())
        rows.append({"id": g["id"], "p@5": float(np.mean(hits)) if hits else 0.0, "base": base})
    return {"precision@5": round(100 * float(np.mean([r["p@5"] for r in rows])), 1),
            "случайно": round(100 * float(np.mean([r["base"] for r in rows])), 1),
            "по запросам": {r["id"]: round(100 * r["p@5"]) for r in rows}}


@torch.no_grad()
def eval_visual(models: dict, proc, E, ok, lids, room, idx, test_ids, k=5):
    """Визуальный судья (eval_visual_judge.judge) на всех 32 golden-запросах, поиск только
    среди объявлений теста: 160 вердиктов на модель вместо 60 у автоматической разметки."""
    from concurrent.futures import ThreadPoolExecutor

    from eval_visual_judge import judge
    from evals import GOLDEN
    test_mask = np.isin(lids, list(test_ids)) & ok
    jobs = []
    for name, m in models.items():
        for g in GOLDEN:
            q = encode(m, proc, [g["ru"]]).cpu().numpy()[0]
            mask = test_mask & (np.isin(room, g["room"]) if g["room"] else True)
            s = np.where(mask, E @ q, -np.inf)
            paths, seen = [], set()
            for i in np.argsort(-s):
                if not np.isfinite(s[i]):
                    break
                if lids[i] not in seen:
                    seen.add(lids[i])
                    paths.append(idx.path.iloc[i])
                    if len(paths) == k:
                        break
            jobs.append((name, g["ru"], paths))
    with ThreadPoolExecutor(6) as ex:
        verdicts = list(ex.map(lambda j: judge(j[1], j[2]), jobs))
    out = {}
    for name in models:
        v = [x for (n, _, _), vs in zip(jobs, verdicts) if n == name for x in vs if x is not None]
        out[name] = {"visual_precision@5": round(100 * float(np.mean(v)), 1), "вердиктов": len(v)}
    return out


def run(epochs=3, eval_only=False, test_fold=0):
    from transformers import AutoModel, AutoProcessor
    torch.manual_seed(SEED)
    idx, E, ok, lids, room, df = load_data()
    targets = listing_targets(E, ok, lids)
    d = df[df.description.fillna("").str.len() >= 40]
    d = d[d.listing_id.astype(str).isin(targets)]
    train_df, test_df = d[d.fold != test_fold], d[d.fold == test_fold]
    tr = [(str(r.listing_id), r.description) for r in train_df.itertuples()]
    te = [(str(r.listing_id), r.description) for r in test_df.itertuples()]
    test_ids = set(df[df.fold == test_fold].listing_id.astype(str))
    print(f"пар для обучения: {len(tr)}, для проверки: {len(te)}, объявлений в тесте: {len(test_ids)}")

    proc = AutoProcessor.from_pretrained(CLIP_MODEL_ID)
    base = AutoModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
    report = {"base": {"desc->photo": eval_desc_to_photo(base, proc, te, targets),
                       "golden": eval_golden(base, proc, E, ok, lids, room, test_ids, df)}}
    print("база:", json.dumps(report["base"], ensure_ascii=False))

    if eval_only and (ADAPTER_DIR / "adapter.pt").exists():
        tuned = attach_adapter(AutoModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval())
    else:
        tuned = train(AutoModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE), proc, tr, targets, epochs=epochs)
        (ADAPTER_DIR / "meta.json").write_text(json.dumps({"r": 16, "epochs": epochs, "model": CLIP_MODEL_ID,
                                                           "targets": TARGETS, "train_pairs": len(tr)}))
    report["lora"] = {"desc->photo": eval_desc_to_photo(tuned, proc, te, targets),
                      "golden": eval_golden(tuned, proc, E, ok, lids, room, test_ids, df)}
    print("LoRA:", json.dumps(report["lora"], ensure_ascii=False))
    vis = eval_visual({"base": base, "lora": tuned}, proc, E, ok, lids, room, idx, test_ids)
    report["base"]["visual"], report["lora"]["visual"] = vis["base"], vis["lora"]
    print("визуальный судья:", json.dumps(vis, ensure_ascii=False))
    b, l = report["base"], report["lora"]
    # Включаем, только если прирост на визуальном судье больше шума: 160 бинарных вердиктов
    # дают стандартную ошибку разницы ≈ 5 п.п., поэтому порог — +5 п.п.
    gain = l["visual"]["visual_precision@5"] - b["visual"]["visual_precision@5"]
    better = gain >= 5 and l["golden"]["precision@5"] >= b["golden"]["precision@5"]
    report["decision"] = (f"включить (USE_TEXT_LORA=1): визуальный судья +{gain:.1f} п.п." if better
                          else f"не включать: прирост {gain:+.1f} п.п. на визуальном судье в пределах шума")
    print("решение:", report["decision"])
    (ART_DIR / "lora_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    run(a.epochs, a.eval_only)
