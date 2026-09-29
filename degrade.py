# -*- coding: utf-8 -*-
"""Порча листа: превращаем чистый PDF в подобие фотокопии.

Настоящие спецификации приходят сканами с перекосом, серым фоном и «грязью».
Чтобы замер что-то значил, лист надо испортить примерно так же, иначе
распознавание меряется на идеальной картинке и цифры получаются красивые,
но бесполезные.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

HERE = Path(__file__).parent
DPI = 300


def degrade(page_img: np.ndarray, angle: float, noise: int, blur: int) -> np.ndarray:
    h, w = page_img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(page_img, m, (w, h), borderValue=255)
    if blur:
        img = cv2.GaussianBlur(img, (blur * 2 + 1, blur * 2 + 1), 0)
    if noise:
        img = np.clip(img.astype(np.int16) +
                      np.random.default_rng(3).integers(-noise, noise, img.shape), 0, 255)
        img = img.astype(np.uint8)
    # лёгкая неравномерность освещения, как у фотографии листа
    grad = np.linspace(0.88, 1.06, w, dtype=np.float32)[None, :]
    return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)


LEVELS = {                  # имя: (угол, шум, размытие)
    "чистый":   (0.0, 0, 0),
    "лёгкий":   (0.4, 8, 0),
    "средний":  (1.1, 18, 1),
    "тяжёлый":  (2.2, 30, 2),
}


def render(pdf: Path, level: str) -> list[np.ndarray]:
    doc = pymupdf.open(pdf)
    out = []
    for page in doc:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(DPI / 72, DPI / 72), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n == 3 else img[:, :, 0]
        out.append(degrade(gray, *LEVELS[level]))
    return out


def save_pdf(pages: list[np.ndarray], path: Path) -> Path:
    """Собрать испорченные страницы обратно в PDF — распознавание работает
    с PDF, как и у заказчика: сканы приходят именно так."""
    doc = pymupdf.open()
    for img in pages:
        ok, buf = cv2.imencode(".png", img)
        h, w = img.shape
        page = doc.new_page(width=w * 72 / DPI, height=h * 72 / DPI)
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(path)
    return path


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "средний"
    pages = render(HERE / "spec_synthetic.pdf", level)
    pdf = save_pdf(pages, HERE / f"scan_{level}.pdf")
    print(f"уровень «{level}»: страниц {len(pages)}, файл {pdf.name}")
