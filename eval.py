# -*- coding: utf-8 -*-
"""Замер: сколько позиций спецификации читается верно и что даёт сверка.

Меряется не «распознавание вообще», а то, ради чего всё делается: сколько
строк попадёт в смету с правильной массой. Отдельно считается, сколько строк
спасла арифметическая проверка «количество × масса единицы = общая масса» —
без неё ошибка распознавания молча уходит в деньги.
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import spec_read as S            # noqa: E402

HERE = Path(__file__).parent
LEVELS = ["чистый", "лёгкий", "средний", "тяжёлый"]


def truth() -> dict[int, dict]:
    out = {}
    for r in csv.DictReader(open(HERE / "truth.csv", encoding="utf-8")):
        out[int(r["поз"])] = {"кол": int(r["кол"]),
                              "масса_ед": float(r["масса_ед"]),
                              "масса_всего": float(r["масса_всего"])}
    return out


def measure(pdf: Path, reader, check: bool, own: bool = False) -> dict:
    """own=True — читать своей моделью (15 МБ) вместо easyocr (полтора
    гигабайта с torch). Ради поставки заказчику важно знать не «работает ли
    распознавание вообще», а сколько теряется при переходе на лёгкую."""
    items = []
    # Страниц в листе столько, сколько их в файле: 48 позиций по 20 строк
    # дают три. Пока здесь стояла двойка, треть позиций молча считалась
    # непрочитанной, и замер занижал сам себя.
    import pymupdf
    pages = pymupdf.open(str(pdf)).page_count
    for page in range(pages):
        gray, angle = S.page_image(str(pdf), page, with_angle=True)
        ys, xs = S.grid(gray)
        got = (S.cells_pdf(str(pdf), page, ys, xs, reader, gray=gray, angle=angle) if own
               else S.cells(gray, ys, xs, reader))
        rows = S.rows(got, len(xs) - 1)
        items += S.verify(rows) if check else rows
    real = truth()
    ok = bad = miss = 0
    for poz, t in real.items():
        got = next((i for i in items if i["поз"] == poz), None)
        m = (got or {}).get("масса_всего")
        if m is None:
            miss += 1
        elif abs(m - t["масса_всего"]) < 0.05:
            ok += 1
        else:
            bad += 1
    return {"позиций": len(real), "верно": ok, "ошибка": bad, "не прочиталось": miss,
            "найдено строк": len(items)}


if __name__ == "__main__":
    own = len(sys.argv) > 1 and sys.argv[1] == "своя"
    if own:
        import recognize
        reader = recognize.Reader()
    else:
        import easyocr
        reader = easyocr.Reader(["ru"], gpu=False, verbose=False)
    print(f"распознаватель: {'своя модель' if own else 'easyocr'}")
    print(f"{'уровень':<10} {'сверка':<8} {'верно':>6} {'ошибка':>7} {'пропуск':>8}")
    for level in LEVELS:
        pdf = HERE / f"scan_{level}.pdf"
        for check in (False, True):
            r = measure(pdf, reader, check, own)
            print(f"{level:<10} {'да' if check else 'нет':<8} "
                  f"{r['верно']:>6} {r['ошибка']:>7} {r['не прочиталось']:>8}")
