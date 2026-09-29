# -*- coding: utf-8 -*-
"""Синтетическая спецификация ГОСТ 2.106: генератор для замера распознавания.

Чертежи заказчиков в открытый репозиторий выкладывать нельзя, поэтому проверка
идёт на сгенерированных листах: та же сетка колонок, те же форматы записи массы
(«11,2/22,4 кг»), те же типы материалов. Зато известна правда до последней
цифры — есть с чем сравнивать.
"""
import csv, random
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
# Ширины колонок в миллиметрах: в сумме 190 при поле A4 в 210 — как в реальной
# спецификации, где рамка отступает от краёв листа.
# Порядок колонок как на настоящем листе: боковая полоса рамки чертежа,
# узкая пустая, формат, зона, позиция, обозначение, наименование, количество,
# примечание. Без боковой полосы разбор не найдёт привычных колонок —
# он ориентируется на их номера.
COLS = [6, 4, 8, 8, 10, 30, 62, 10, 24]
HEAD = ["", "", "Ф", "З", "Поз", "Обозначение", "Наименование", "Кол", "Прим."]
NAMES = ["Ребро", "Косынка", "Планка", "Стойка", "Кронштейн", "Опора", "Ригель",
         "Связь", "Лист опорный", "Подкладка", "Упор", "Распорка", "Стяжка", "Платик"]
SHEETS = [("Лист {t} ГОСТ 19903-2015", "Ст3 ГОСТ 14637-89"),
          ("Лист {t} ГОСТ 19903-2015", "09Г2С ГОСТ 19281-2014")]
THICK = [4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50]


def rows(n: int, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        qty = rnd.choice([1, 1, 2, 2, 4, 6, 8, 10, 14, 22])
        one = round(rnd.uniform(0.8, 180), 1)
        t = rnd.choice(THICK)
        pat, mark = rnd.choice(SHEETS)
        out.append({
            "поз": i,
            "обозначение": f"2-{rnd.randint(1000, 9999)}-0.00.{i:02d}",
            "наименование": rnd.choice(NAMES),
            "материал": f"{pat.format(t=t)}; {mark}",
            "толщина": t,
            "кол": qty,
            "масса_ед": one,
            "масса_всего": round(one * qty, 1),
        })
    return out


def draw(items: list[dict], path: Path, rows_per_page: int = 20) -> Path:
    """Лист спецификации: под каждую позицию две строки — наименование и,
    ниже, материал с сортаментом. Так это и выглядит по ГОСТ 2.106, и так же
    ведёт себя разбор: строка без номера позиции подклеивается к предыдущей.
    """
    doc = pymupdf.open()
    mm = 72 / 25.4
    font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    for start in range(0, len(items), rows_per_page):
        page = doc.new_page(width=210 * mm, height=297 * mm)
        x0, y0, rh = 10 * mm, 20 * mm, 7 * mm
        xs = [x0]
        for w in COLS:
            xs.append(xs[-1] + w * mm)
        chunk = items[start:start + rows_per_page]
        lines = len(chunk) * 2 + 1          # по две строки на позицию плюс шапка
        for r in range(lines + 1):
            y = y0 + r * rh
            page.draw_line((xs[0], y), (xs[-1], y), width=0.6)
        for x in xs:
            page.draw_line((x, y0), (x, y0 + lines * rh), width=0.6)
        for c, name in enumerate(HEAD):
            page.insert_text((xs[c] + 2, y0 + rh * 0.72), name, fontsize=6.5,
                             fontname="cyr", fontfile=font)
        for i, it in enumerate(chunk):
            y_name = y0 + (i * 2 + 1) * rh + rh * 0.72
            y_mat = y_name + rh
            first = ["", "", "А4", "", str(it["поз"]), it["обозначение"],
                     it["наименование"], str(it["кол"]),
                     f'{it["масса_ед"]:g}/{it["масса_всего"]:g} кг'.replace(".", ",")]
            second = ["", "", "", "", "", "", it["материал"], "", ""]
            for cells, y in ((first, y_name), (second, y_mat)):
                for c, text in enumerate(cells):
                    if text:
                        page.insert_text((xs[c] + 2, y), text, fontsize=6.5,
                                         fontname="cyr", fontfile=font)
    doc.save(path)
    return path


if __name__ == "__main__":
    items = rows(48)
    pdf = draw(items, HERE / "spec_synthetic.pdf")
    with open(HERE / "truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    print(f"сгенерировано позиций: {len(items)}")
    print(f"суммарная масса: {round(sum(i['масса_всего'] for i in items), 1)} кг")
    print(f"файлы: {pdf.name}, truth.csv")
