#!/usr/bin/env python3
"""Единая точка распознавания: прямоугольник на листе — строка текста.

Работает через onnxruntime и файл `model.onnx` рядом с программой. Это выбрано
ради установки у заказчика: тот же самый распознаватель в PyTorch потребовал бы
двух гигабайт и прав администратора, а здесь — пятнадцать мегабайт модели и
библиотека на двенадцать. Если ONNX рядом не оказалось (например, у нас во время
обучения), берётся модель PyTorch — результат тот же.

Вырезка делается не из готового растра страницы, а отдельным рендером именно
этого прямоугольника с большим увеличением. Разница принципиальная: текст на
листе А3 при рендере целой страницы размазывается в кашу, а вырезкой он
получается таким, будто снят вблизи.
"""
import io
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
IMG_H, IMG_W = 64, 600
ZOOM = 5.0
ONNX = ROOT / "model.onnx"
CHARS = ROOT / "model.chars.json"
WEIGHTS = ROOT / "finetuned.pth"


class Reader:
    MAX_PX = 6000          # сторона вырезки, дальше рендер только тратит память

    def __init__(self, onnx=ONNX, weights=WEIGHTS):
        self.session = self.torch_model = None
        if Path(onnx).exists() and CHARS.exists():
            import onnxruntime
            self.session = onnxruntime.InferenceSession(
                str(onnx), providers=["CPUExecutionProvider"])
            self.chars = json.loads(CHARS.read_text(encoding="utf-8"))
        else:
            # Запасной путь — для обучения у меня; у заказчика рядом лежит
            # model.onnx, и сюда не заходят. Но если файл потеряется при
            # распаковке, человек должен увидеть внятную причину, а не
            # «ModuleNotFoundError: torch».
            try:
                import torch
            except ImportError:
                raise SystemExit(
                    "Рядом с программой нет файла model.onnx — распознавать нечем.\n"
                    "Распакуйте папку целиком: модель лежит в ней файлом на 15 МБ.")
            sys.path.insert(0, str(ROOT))
            from finetune import build
            model, conv, _ = build()
            if Path(weights).exists():
                model.load_state_dict(torch.load(weights, map_location="cpu"))
            model.eval()
            self.torch, self.torch_model = torch, model
            self.chars = list(conv.character)

    # ---------- картинка ----------

    def crop(self, page, rect, zoom=ZOOM):
        """Вырезка с увеличением, с оглядкой на её размер.

        Ячейка может оказаться и в пол-листа А1 — у такой при пятикратном
        увеличении рендер уходит в сотни мегапикселей и падает. Увеличение для
        таких ячеек снижается ровно настолько, чтобы влезть; мелкие, ради
        которых всё и затевалось, увеличиваются как обычно.
        """
        r = fitz.Rect(rect) & page.rect
        if r.is_empty or r.width < 1 or r.height < 1:
            return Image.new("L", (8, 8), 255)
        z = min(zoom, self.MAX_PX / max(r.width, r.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=r)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def _prep(self, img):
        w = max(8, min(IMG_W, int(img.width * IMG_H / max(img.height, 1))))
        im = img.convert("L").resize((w, IMG_H), Image.LANCZOS)
        canvas = Image.new("L", (IMG_W, IMG_H), 255)
        canvas.paste(im, (0, 0))
        a = np.asarray(canvas, dtype=np.float32) / 255.0
        return ((a - 0.5) / 0.5)[None, None, :, :]

    # ---------- распознавание ----------

    def read_image(self, img):
        x = self._prep(img)
        if self.session is not None:
            logits = self.session.run(None, {"image": x})[0]
            idx = logits.argmax(axis=2).reshape(-1)
        else:
            t = self.torch.from_numpy(x)
            with self.torch.no_grad():
                logits = self.torch_model(t, None)
            idx = logits.max(2)[1].cpu().numpy().reshape(-1)
        # Обычное жадное CTC-декодирование: подряд идущие одинаковые классы —
        # один знак, нулевой класс — пустота между знаками.
        out, prev = [], -1
        for i in idx:
            i = int(i)
            if i != prev and i != 0:
                out.append(self.chars[i])
            prev = i
        return "".join(out).strip()

    def read_image_wide(self, img):
        """Длинная строка — по кускам.

        Модель принимает картинку 600 × 64, то есть строку не длиннее девяти с
        половиной высот. Подпись трубы на профиле тянется через весь лист и в
        полтора десятка раз длиннее: при сжатии до входного размера буквы
        сплющиваются, и вместо «Труба КОРСИС SN8 DN/OD 250» выходит каша.
        Поэтому такая строка режется по пробелам между словами на куски
        подходящей длины и склеивается обратно.
        """
        ratio = img.width / max(img.height, 1)
        limit = IMG_W / IMG_H
        if ratio <= limit * 1.15:
            return self.read_image(img)

        a = np.asarray(img.convert("L"))
        ink = (a < 160).sum(axis=0)
        gaps, run = [], None                     # пустые полосы между словами
        for x, v in enumerate(ink):
            if v == 0 and run is None:
                run = x
            elif v > 0 and run is not None:
                gaps.append((run, x)); run = None
        if run is not None:
            gaps.append((run, len(ink)))

        step = int(img.height * limit)            # сколько влезает в один кусок
        parts, start = [], 0
        while img.width - start > step:
            # режем по самому широкому просвету в пределах куска, а если
            # просветов нет вовсе — ровно по границе, хуже уже не будет
            here = [g for g in gaps if start + step * 0.4 < (g[0] + g[1]) / 2 < start + step]
            cut = max(here, key=lambda g: g[1] - g[0], default=None)
            end = int((cut[0] + cut[1]) / 2) if cut else start + step
            parts.append((start, end))
            start = end
        parts.append((start, img.width))

        out = []
        for x0, x1 in parts:
            if x1 - x0 < 6:
                continue
            # Пустой кусок читать нельзя: на чистом поле модель всё равно что-то
            # выдаёт, и к «51,72» приписывается лишняя цифра. В числовой ячейке
            # такая приписка превращается в другое число, и это хуже, чем
            # не прочитать вовсе.
            if ink[x0:x1].sum() < img.height * 0.05:
                continue
            t = self.read_image(img.crop((x0, 0, x1, img.height)))
            if t:
                out.append(t)
        return " ".join(out)

    MIN_INK = 0.0008        # доля тёмных точек, ниже которой ячейка считается пустой

    # Буквы, цифры и знаки, из которых состоит нормальная подпись на чертеже.
    GOOD = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
               "0123456789 .,:;-–—/\\()[]%№×хХ+=°⌀")

    @classmethod
    def readable(cls, text, floor=0.75):
        """Осмысленный ли это текст, а не мусор из сломанной кодировки.

        Часть чертежей выходит из AutoCAD с текстовым слоем в нестандартной
        кодировке: символы в файле есть, но читаются как «����». Брать такой
        слой нельзя — он выглядит как текст и молча подменяет распознавание.
        Проверяем долю знаков, из которых вообще состоят подписи на чертежах.
        """
        t = (text or "").strip()
        if not t:
            return False
        good = sum(1 for c in t if c in cls.GOOD)
        return good / len(t) >= floor

    @staticmethod
    def text_layer(page, rect):
        """Текст из самого PDF, если он там есть.

        Часть разделов выпускается с настоящим текстовым слоем — в них буквы не
        переведены в кривые. Распознавать такой лист бессмысленно и вредно:
        текст уже есть, он точен, и берётся мгновенно. Координаты приходят в
        экранной системе, поэтому перед выборкой их надо вернуть в систему
        страницы обратным поворотом.
        """
        if not page.get_text("text").strip():
            return None
        try:
            box = fitz.Rect(rect) * ~page.rotation_matrix
            got = page.get_textbox(box)
        except Exception:                      # noqa: BLE001
            return None
        got = " ".join(got.split())
        if not got or not Reader.readable(got):
            return None
        return got

    def read(self, page, rect, zoom=ZOOM):
        if rect.width < 2 or rect.height < 2:
            return ""
        direct = self.text_layer(page, rect)
        if direct is not None:
            return direct
        img = self.crop(page, rect, zoom)
        # Пустую ячейку модели показывать нельзя: она обучена всегда что-то
        # отвечать и на чистом поле выдаёт «8» или «с». В таблице профиля пустых
        # ячеек больше половины, и каждый такой ответ — выдуманный участок.
        a = np.asarray(img.convert("L"))
        if (a < 160).mean() < self.MIN_INK:
            return ""
        # Узкая и высокая ячейка — это подпись, набранная снизу вверх: так
        # подписывают колонки таблиц колодцев и отметки над профилем. Модель
        # читает только горизонтальные строки, поэтому такую вырезку сначала
        # кладём набок.
        if img.height > img.width * 1.6:
            img = img.transpose(Image.ROTATE_270)
        return self.read_image_wide(img)

    def read_lines(self, page, rect, zoom=ZOOM):
        """Многострочная ячейка: каждая строка распознаётся отдельно.

        Модель обучена на одной строке текста; если подать ей две, она сминает
        их в одну кашу. Строки разделяются по горизонтальным просветам в чернилах
        самой вырезки — это надёжнее, чем делить прямоугольник поровну: в ячейке
        бывает и одна строка, и три, и высота их разная.
        """
        direct = self.text_layer(page, rect)
        if direct is not None:
            return [direct]
        img = self.crop(page, rect, zoom)
        if img.height > img.width * 1.6:          # подпись набрана снизу вверх
            img = img.transpose(Image.ROTATE_270)
        a = np.asarray(img.convert("L"))
        dark = (a < 160).sum(axis=1)
        # Межстрочный промежуток не всегда пуст: хвосты «р» и «Д» с верхней
        # строки заходят в него, и деление по строгому нулю склеивает две
        # строки в одну — модель выдаёт кашу вместо «Труба ПЭ100 … SDR11-32х3,0».
        # Поэтому строкой считается то, что выше десятой доли самой плотной
        # строки пикселей: настоящий промежуток всё равно много светлее текста.
        floor = max(1, int(dark.max() * 0.1)) if dark.size else 1
        rows, run = [], None
        for y, v in enumerate(dark):
            if v > floor and run is None:
                run = y
            elif v <= floor and run is not None:
                rows.append((run, y)); run = None
        if run is not None:
            rows.append((run, len(dark)))
        if not rows:
            return []
        # Склеивать полосы нужно осторожно. Надстрочный элемент — крышка «й»,
        # точки «ё» — это тонкая полоска над строкой, и её надо вернуть своей
        # строке. Но две полноценные строки текста разделяет промежуток, который
        # у плотной вёрстки бывает меньше половины их высоты: по прежнему правилу
        # они слипались, и «Труба КОРСИС SN8 DN/OD 315 / ТУ 22.21.21-001-…»
        # читалось как одна каша. Поэтому сливаем, только если одна из полос
        # заметно ниже остальных — то есть это и есть надстрочный элемент.
        heights = [b - a_ for a_, b in rows]
        med = np.median(heights) or 1
        merged = [list(rows[0])]
        for (a_, b), h in zip(rows[1:], heights[1:]):
            prev_h = merged[-1][1] - merged[-1][0]
            small = min(h, prev_h) < med * 0.45
            if small and a_ - merged[-1][1] < med * 0.5:
                merged[-1][1] = b
            else:
                merged.append([a_, b])
        # Одна строка — читаем вырезку как есть. Обрезка вплотную по чернилам
        # заметно портит результат: модель обучена на ячейках с полями, и без
        # них «1. Труба КОРСИС SN8 DN/OD 160» превращается в кашу. Режем только
        # то, что действительно состоит из нескольких строк.
        if len(merged) < 2:
            t = self.read_image_wide(img)
            return [t] if t else []
        pad = max(2, int(np.median([b - a_ for a_, b in merged]) * 0.25))
        out = []
        for a_, b in merged:
            if b - a_ < 4:
                continue
            line = img.crop((0, max(0, a_ - pad), img.width, min(img.height, b + pad)))
            t = self.read_image_wide(line)
            if t:
                out.append(t)
        return out


_reader = None


def reader():
    global _reader
    if _reader is None:
        _reader = Reader()
    return _reader


if __name__ == "__main__":
    import os
    from profiles import profile_rows
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Downloads/2137.19.8-00-НВК4 вер.0А.pdf")
    doc = fitz.open(src)
    rd = reader()
    print("движок:", "onnxruntime" if rd.session is not None else "pytorch")
    page = doc[5]
    for y0, y1, lab, cells in profile_rows(page):
        print("  ", " ".join(rd.read_lines(page, lab)) if lab else "—")
