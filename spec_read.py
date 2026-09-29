# -*- coding: utf-8 -*-
"""Спецификация ГОСТ 2.106 со скана: сетка по линиям таблицы + распознавание ячеек.

В машиностроительной спецификации колонки другие, чем в строительной:
Формат | Зона | Поз. | Обозначение | Наименование | Кол. | Примечание.
Вектора в присланных файлах нет — это фотокопии, поэтому сетку ищем
морфологией по изображению, а не по чертёжным примитивам.

Масса детали стоит в примечании в виде «330/660 кг» — масса единицы и общая.
"""
import re, sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

DPI = 300
# Косую черту в «3/24кг» распознаватель часто теряет или превращает в пробел,
# а «кг» читает как «к2», «ке», «кг.». Поэтому разделителем считаем что угодно
# между двумя числами, но требуем, чтобы дальше стояла единица массы —
# иначе под шаблон попадут размеры заготовки вида «400х315».
UNIT = r"\s*[кk][гг2еzаa]"   # «кг» читается как «к2», «ке», «ка», «кz»
# Между массой единицы и общей может стоять что угодно: косая, пробел, скобка,
# запятая — распознаватели портят разделитель по-разному. Поэтому допускаем
# до четырёх любых нецифровых знаков, но требуем единицу массы следом:
# иначе под шаблон попадут размеры заготовки «400х315 мм».
MASS = re.compile(rf"(\d+[.,]?\d*)\D{{1,4}}(\d+[.,]?\d*){UNIT}", re.I)
ONE_MASS = re.compile(rf"^(\d+[.,]?\d*){UNIT}", re.I)


def page_image(pdf: str, page_no: int, with_angle: bool = False):
    """Растр страницы, выровненный по перекосу.

    with_angle=True — вернуть ещё и угол поворота: он нужен тому, кто режет
    ячейки не из этого растра, а заново из PDF. Прямоугольники ячеек найдены
    на выровненной картинке, и в неповёрнутом PDF они попадут мимо текста.
    """
    doc = pymupdf.open(pdf)
    pix = doc[page_no].get_pixmap(matrix=pymupdf.Matrix(DPI / 72, DPI / 72), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n == 3 else img[:, :, 0]
    # Выравниваем сразу при загрузке: если повернуть только для поиска сетки,
    # координаты ячеек перестанут совпадать с содержимым страницы.
    angle = skew_angle(gray)
    out = rotate(gray, angle) if angle else gray
    return (out, angle) if with_angle else out


def skew_angle(gray: np.ndarray, limit: float = 4.0) -> float:
    """Угол перекоса листа по самым длинным прямым; 0, если лист ровный.

    Сетку мы ищем морфологией: горизонтальные линии — длинным горизонтальным
    ядром. На скане с наклоном в полтора градуса такая линия перестаёт быть
    горизонтальной, ядро её не находит, и таблица «исчезает» целиком.
    Проверено на синтетике: при наклоне 1,1° разбор давал ноль позиций.
    """
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=200,
                            minLineLength=gray.shape[1] // 4, maxLineGap=12)
    if lines is None:
        return 0.0
    angles = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in line)
        if x2 == x1:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) <= limit:            # берём только почти горизонтальные
            angles.append(a)
    if not angles:
        return 0.0
    angle = float(np.median(angles))
    return 0.0 if abs(angle) < 0.15 else angle   # ровный лист не трогаем


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(gray: np.ndarray, limit: float = 4.0) -> np.ndarray:
    """Выровненный растр (оставлено для вызовов со стороны)."""
    a = skew_angle(gray, limit)
    return rotate(gray, a) if a else gray


def grid(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Координаты линий таблицы. Порог — по Оцу, линии — морфологией."""
    # Фотокопия почти всегда освещена неравномерно: один край листа светлее.
    # На светлой стороне линии бледнее, и морфология их теряет — на проверочном
    # скане пропадали три правых колонки. CLAHE выравнивает контраст по клеткам,
    # после него линии одинаково видны по всему листу.
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    h, w = bw.shape
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1)))
    # Ядро для вертикалей короче, чем для горизонталей: на размытом скане
    # тонкая линия колонки прерывается, и длинное ядро её не собирает —
    # правые колонки листа пропадали целиком.
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 45)))

    def lines(mask, axis):
        prof = mask.sum(axis=axis) / 255
        # Порог берём от самой длинной найденной линии. На размытом скане
        # тонкие линии слабее жирной рамки, и при пороге 0,4 часть колонок
        # пропадала: на проверочном листе находилось 7 вертикальных вместо 10,
        # колонки съезжали, и разбор давал ноль позиций.
        thresh = prof.max() * 0.22
        idx = np.where(prof > thresh)[0]
        out, group = [], []
        for i in idx:
            if group and i - group[-1] > 5:
                out.append(int(np.mean(group))); group = []
            group.append(i)
        if group:
            out.append(int(np.mean(group)))
        return out

    return lines(horiz, 1), lines(vert, 0)      # y-линии, x-линии


def cells(gray, ys, xs, reader, min_h=14, min_w=14):
    """Текст непустых ячеек: {(строка, колонка): текст}."""
    out = {}
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            y0, y1, x0, x1 = ys[r] + 2, ys[r + 1] - 2, xs[c] + 2, xs[c + 1] - 2
            if y1 - y0 < min_h or x1 - x0 < min_w:
                continue
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0 or (crop < 128).mean() < 0.01:   # пусто
                continue
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res = reader.readtext(big, detail=1, paragraph=False)
            txt = " ".join(t for _, t, cf in sorted(res, key=lambda z: z[0][0][0]) if cf > 0.25)
            if txt.strip():
                out[(r, c)] = txt.strip()
    return out


# Распознаватель путает цифры с похожими буквами: «115/230кг» превращается
# в «115/23Ок2», «23/46» в «23/4б», «13» в «1З». В примечании спецификации
# букв, кроме единицы массы, быть не может, поэтому подмену можно откатить.
LOOKALIKE = str.maketrans({"О": "0", "о": "0", "O": "0", "o": "0",
                           "З": "3", "з": "3", "б": "6", "Б": "6",
                           "l": "1", "I": "1", "|": "1", "S": "5"})


def fix_digits(text: str) -> str:
    """Буквы-двойники обратно в цифры — только там, где рядом стоят цифры."""
    out = []
    for i, ch in enumerate(text):
        near = (text[i - 1:i] + text[i + 1:i + 2])
        if ord(ch) in LOOKALIKE and any(c.isdigit() for c in near):
            out.append(ch.translate(LOOKALIKE))
        else:
            out.append(ch)
    return "".join(out)


def cells_pdf(pdf: str, page_no: int, ys, xs, reader, min_h=14, min_w=14,
              gray=None, angle: float = 0.0):
    """То же, что cells(), но ячейка рендерится прямо из PDF с увеличением.

    Так читает наша модель: вырезка из страницы в большом разрешении вместо
    куска общего растра. Для поставки заказчику это единственный путь —
    easyocr тянет за собой torch на полтора гигабайта, а здесь модель
    весит пятнадцать мегабайт.
    """
    import pymupdf
    from PIL import Image
    page = pymupdf.open(pdf)[page_no]
    k = DPI / 72
    out = {}
    # Лист с перекосом резать из PDF нельзя: прямоугольники ячеек найдены на
    # выровненной картинке, а страница в файле осталась повёрнутой, и вырезка
    # приходит мимо текста. На проверочном скане с наклоном 2,2° так терялись
    # все 48 значений. Поэтому у наклонного листа ячейки берутся из того же
    # выровненного растра, а увеличение делается растяжением.
    if angle and gray is not None:
        for r in range(len(ys) - 1):
            for c in range(len(xs) - 1):
                y0, y1 = ys[r] + 3, ys[r + 1] - 3
                x0, x1 = xs[c] + 3, xs[c + 1] - 3
                if y1 - y0 < min_h or x1 - x0 < min_w:
                    continue
                crop = gray[y0:y1, x0:x1]
                if crop.size == 0 or (crop < 128).mean() < 0.01:
                    continue
                img = Image.fromarray(crop).convert("L")
                img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
                txt = (reader.read_image_wide(img) if img.width > img.height * 8
                       else reader.read_image(img)).strip()
                if txt:
                    out[(r, c)] = txt
        return out
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            if ys[r + 1] - ys[r] < min_h or xs[c + 1] - xs[c] < min_w:
                continue
            # Отступаем внутрь от линий таблицы. Без этого в вырезку попадает
            # сама граница ячейки, и модель читает её как скобку: «1» из
            # колонки позиции превращается в «(1», строка перестаёт быть
            # номером, и позиция теряется целиком. На чистом листе так
            # терялось 44 значения из 48.
            pad = 3
            rect = ((xs[c] + pad) / k, (ys[r] + pad) / k,
                    (xs[c + 1] - pad) / k, (ys[r + 1] - pad) / k)
            img = reader.crop(page, rect)
            # На пустой ячейке распознаватель уверенно выдаёт «1» или «16» —
            # это не текст, а разводы бумаги. Пустоту отсекаем до модели:
            # доля тёмных точек в вырезке меньше процента.
            a = np.asarray(img.convert("L"))
            if a.size == 0 or (a < 128).mean() < 0.01:
                continue
            txt = (reader.read_image_wide(img) if img.width > img.height * 8
                   else reader.read_image(img)).strip()
            if txt:
                out[(r, c)] = txt
    return out


def mass_of(note: str) -> tuple[float | None, float | None]:
    """«330/660 кг» -> (330.0, 660.0); «107кг» -> (107.0, None)."""
    if not note:
        return None, None
    note = fix_digits(note)
    m = MASS.search(note)
    if m:
        return float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
    one = ONE_MASS.match(note.strip())
    return (float(one.group(1).replace(",", ".")), None) if one else (None, None)


# Колонки, как они идут в спецификации ГОСТ 2.106 слева направо. Нулевая —
# боковая полоса рамки чертежа, данных в ней нет.
# Проверено на листах рамы кантователя 29.09.2026: нулевая колонка — боковая
# полоса рамки, первая пустая, дальше формат, зона, позиция, обозначение,
# наименование, количество, примечание.
# Отсчёт идёт от правого края таблицы, а не от левого. Слева у листа полосы
# рамки чертежа, и на размытом или перекошенном скане две близкие линии рамки
# сливаются в одну: колонок находится на одну меньше, все номера съезжают
# влево, в «количестве» оказывается наименование, и лист разбирается в ноль
# позиций. Проверено на синтетике с наклоном 1,1° — до этой правки средний
# уровень порчи давал 0 позиций из 48 при том, что более тяжёлый давал 33.
# Правый край таблицы совпадает с краем рамки и находится всегда.
ORDER = ["примечание", "кол", "наименование", "обозначение", "поз", "зона", "формат"]


def columns(ncols: int) -> dict[str, int]:
    """Номера колонок при данном их числе на листе, отсчёт справа налево."""
    return {name: ncols - 1 - i for i, name in enumerate(ORDER)}


def rows(got: dict, ncols: int) -> list[dict]:
    """Строки спецификации. Позиция стоит в своей колонке, но у многострочных
    записей (материал, сортамент, длина) она пустая — такие строки
    подклеиваются к предыдущей позиции, а не теряются."""
    COLS = columns(ncols)
    items, cur = [], None
    for r in sorted({k[0] for k in got}):
        row = {c: got[(r, c)] for (rr, c) in got if rr == r}
        poz = row.get(COLS["поз"], "").strip(" .")
        name = row.get(COLS["наименование"], "").strip()
        # Номер позиции — одно-два цифры подряд и ничего больше. Своя модель
        # на разводах бумаги выдаёт «1» и «16» в служебных колонках, и без
        # этой проверки каждая такая клякса открывает новую позицию.
        if poz.isdigit() and 0 < int(poz) < 200 and len(poz) <= 3 \
           and (row.get(COLS["наименование"], "").strip() or row.get(COLS["обозначение"], "").strip()):
            if cur:
                items.append(cur)
            m1, m2 = mass_of(row.get(COLS["примечание"], ""))
            cur = {"поз": int(poz), "строка": r,
                   "обозначение": row.get(COLS["обозначение"], "").strip(),
                   "наименование": name, "кол": row.get(COLS["кол"], "").strip(),
                   "примечание": row.get(COLS["примечание"], "").strip(),
                   "масса_ед": m1, "масса_всего": m2, "материал": []}
        elif cur is not None:
            if name:
                cur["материал"].append(name)
            if not cur["кол"] and row.get(COLS["кол"], "").strip():
                cur["кол"] = row[COLS["кол"]].strip()
            if not cur["масса_ед"]:
                m1, m2 = mass_of(row.get(COLS["примечание"], ""))
                cur["масса_ед"], cur["масса_всего"] = m1, m2
    if cur:
        items.append(cur)
    for it in items:
        it["материал"] = "; ".join(it["материал"])
        if it["масса_ед"] is None:          # масса встречается и в строке размеров
            m1, m2 = mass_of(it["материал"])
            it["масса_ед"], it["масса_всего"] = m1, m2
    return items


def verify(items: list[dict]) -> list[dict]:
    """Массу единицы и общую массу распознаёт один и тот же OCR, и он ошибается:
    «6/132» читается как «6/432», «115/230» как «115/23». Но эти три числа
    связаны: количество × масса единицы должно давать общую массу. Там, где
    не сходится, доверяем произведению и помечаем строку — потерянный ноль
    в итоге дороже, чем лишняя пометка.
    """
    for it in items:
        qty = int(re.sub(r"\D", "", it["кол"]) or 0) or None
        it["кол_число"] = qty
        m1, m2 = it["масса_ед"], it["масса_всего"]
        it["масса_пересчитана"] = ""
        if qty and m1:
            calc = round(qty * m1, 2)
            if m2 is None:
                it["масса_всего"], it["масса_пересчитана"] = calc, "общая масса не прочиталась"
            elif abs(calc - m2) > max(0.05, calc * 0.01):
                # Не сошлось — ошиблось одно из трёх чисел, и не обязательно
                # масса. Если общая масса делится на массу единицы нацело,
                # то обе массы согласованы между собой, а неверно прочиталось
                # количество: «12» вместо «2», «4» вместо «14». Тогда массы
                # трогать нельзя — иначе в смету уйдёт вес, которого в
                # чертеже нет.
                k = m2 / m1 if m1 else 0
                n = round(k)
                if 0 < n < 200 and n != qty and abs(k - n) <= max(0.02, n * 0.01):
                    it["кол_число"] = n
                    it["масса_пересчитана"] = f"количество: в чертеже {qty}, по массам {n}"
                else:
                    it["масса_пересчитана"] = f"в чертеже {m2:g}, по количеству {calc:g}"
                    it["масса_всего"] = calc
        elif qty is None and m1 and m2:
            it["кол_число"] = round(m2 / m1) if m1 else None
            it["масса_пересчитана"] = "количество восстановлено из масс"
    return items


def read_page(pdf: str, page_no: int, reader, second=True, own=False) -> list[dict]:
    """Лист спецификации.

    Основное чтение — easyocr: он лучше на словах. Если у строки масса не
    прочиталась или не сошлась с количеством, эта же ячейка перечитывается
    своей моделью, дообученной на чертёжном шрифте: две модели ошибаются
    по-разному, и арифметика показывает, кто прав. Перечитываются единицы
    строк, поэтому лишнего времени это почти не стоит.

    own=True — читать всё своей моделью и не трогать easyocr вовсе. Это режим
    установки у заказчика: easyocr тянет за собой torch на полтора гигабайта,
    а своя модель весит пятнадцать мегабайт. По замеру на синтетике она
    отстаёт на середине шкалы порчи, но ошибается не чаще: чаще молчит, и
    непрочитанная строка помечается в отчёте, а не уходит в цену.
    """
    gray, angle = page_image(pdf, page_no, with_angle=True)
    ys, xs = grid(gray)
    got = (cells_pdf(pdf, page_no, ys, xs, reader, gray=gray, angle=angle) if own
           else cells(gray, ys, xs, reader))
    items = verify(rows(got, len(xs) - 1))
    if not second or own:      # своей моделью уже прочитано, второго мнения нет
        return items

    COLS = columns(len(xs) - 1)
    doubtful = [it for it in items if it.get("масса_пересчитана") or not it.get("масса_всего")]
    if not doubtful:
        return items

    from dual import reread_row, pick_mass, qty_of
    for it in doubtful:
        r = it.get("строка")
        if r is None:
            continue
        # Заглядывать в соседние строки таблицы нельзя: у позиций с материалом
        # запись растянута на несколько строк, и снизу стоит масса уже другой
        # позиции. Проверено 29.09: такая «помощь» дала позиции 15 массу
        # соседа. Лучше оставить строку непрочитанной и пометить её.
        # Колонки передаём явно: их номера зависят от того, сколько линий
        # нашлось на листе, и жёстко зашитая пара сместилась бы вместе с ними.
        alt = reread_row(pdf, page_no, ys, xs, r,
                         cols=(COLS["кол"], COLS["примечание"]))
        qty = qty_of([it.get("кол", ""), alt.get(COLS["кол"], "")])
        m1, m2, why = pick_mass([it.get("примечание", ""), alt.get(COLS["примечание"], ""),
                                 it.get("материал", "")], qty)
        if m1:
            it["кол_число"] = qty or it.get("кол_число")
            it["масса_ед"], it["масса_всего"] = m1, m2
            it["масса_пересчитана"] = "" if why.startswith("сошлось") else why
    return items


def read_spec(pdf: str, pages: range, reader, own: bool = False) -> list[dict]:
    """Спецификация целиком: листы идут подряд, нумерация позиций сквозная."""
    items = []
    for p in pages:
        items += read_page(pdf, p, reader, own=own)
    return items


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else \
        "materials/2-2141-0.00.00 Рама кантователя Vai Cosim (1) (1).pdf"
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    import easyocr
    reader = easyocr.Reader(["ru"], gpu=False, verbose=False)
    items = read_page(pdf, page, reader)
    print(f"лист {page + 1}: позиций {len(items)}")
    for it in items:
        mass = f"{it['масса_ед']}/{it['масса_всего']}" if it["масса_ед"] else "—"
        flag = f"  ⚠ {it['масса_пересчитана']}" if it["масса_пересчитана"] else ""
        print(f"  {it['поз']:>3} {it['наименование'][:26]:<28} кол {str(it['кол_число']):<4} масса {mass:<14}{flag}")
