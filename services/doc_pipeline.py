import io
import logging
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageOps

from config import settings
from models import RawSource
from services.gemini_client import GeminiParseError, gemini_client

logger = logging.getLogger(__name__)

_CHARS_PER_PAGE_THRESHOLD = 100

_ocr_typed_prompt = (
    Path(__file__).parent.parent / "prompts" / "ocr_typed.txt"
).read_text()
_ocr_prescription_prompt = (
    Path(__file__).parent.parent / "prompts" / "ocr_prescription.txt"
).read_text()


def detect_kind(filename: str, mime: str, first_bytes: bytes) -> str:
    mime_lower = mime.lower()
    name_lower = (filename or "").lower()
    if (
        "pdf" in mime_lower
        or name_lower.endswith(".pdf")
        or first_bytes[:4] == b"%PDF"
    ):
        return "pdf_typed"
    return "image_handwritten"


def _preprocess_image(raw: bytes) -> tuple[bytes, str]:
    """Auto-orient, grayscale, gentle contrast. Never hard-binarize."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), "image/jpeg"


def _pil_to_jpeg(pil_img: Image.Image) -> bytes:
    """Convert a rasterized PDF page to preprocessed JPEG bytes."""
    gray = pil_img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    buf = io.BytesIO()
    gray.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _flatten_typed(data: dict) -> str:
    lines: list[str] = []
    if data.get("document_type"):
        lines.append(f"DOCUMENT TYPE: {data['document_type']}")
    if data.get("report_date"):
        lines.append(f"DATE: {data['report_date']}")
    if data.get("ordering_doctor"):
        lines.append(f"DOCTOR: {data['ordering_doctor']}")
    if data.get("clinic_or_hospital"):
        lines.append(f"FACILITY: {data['clinic_or_hospital']}")
    tests = data.get("tests") or []
    if tests:
        lines.append("\nTEST RESULTS:")
        for t in tests:
            flag_str = f" [{t.get('flag')}]" if t.get("flag") else ""
            ref_str = f" [ref: {t['reference_range']}]" if t.get("reference_range") else ""
            unit_str = f" {t['unit']}" if t.get("unit") else ""
            lines.append(f"  {t.get('name', '?')}: {t.get('value', '?')}{unit_str}{ref_str}{flag_str}")
    diagnoses = data.get("diagnoses") or []
    if diagnoses:
        lines.append("\nDIAGNOSES:")
        for d in diagnoses:
            lines.append(f"  - {d}")
    if data.get("summary_impression"):
        lines.append(f"\nIMPRESSION:\n{data['summary_impression']}")
    if data.get("notes"):
        lines.append(f"\nNOTES:\n{data['notes']}")
    return "\n".join(lines)


def _flatten_prescription(data: dict) -> str:
    lines: list[str] = ["PRESCRIPTION"]
    if data.get("prescribed_date"):
        lines.append(f"DATE: {data['prescribed_date']}")
    if data.get("patient_name"):
        lines.append(f"PATIENT: {data['patient_name']}")
    if data.get("patient_dob"):
        lines.append(f"DOB: {data['patient_dob']}")
    if data.get("prescriber_name"):
        lines.append(f"PRESCRIBER: {data['prescriber_name']}")
    if data.get("prescriber_clinic"):
        lines.append(f"CLINIC: {data['prescriber_clinic']}")
    meds = data.get("medications") or []
    if meds:
        lines.append("\nMEDICATIONS:")
        for i, m in enumerate(meds, 1):
            name = m.get("drug_name") or "(unreadable)"
            strength = f" {m['strength']}" if m.get("strength") else ""
            form = f" ({m['form']})" if m.get("form") else ""
            lines.append(f"  {i}. {name}{strength}{form}")
            if m.get("dosage_instructions"):
                lines.append(f"     Sig: {m['dosage_instructions']}")
            if m.get("quantity"):
                lines.append(f"     Qty: {m['quantity']}")
            if m.get("refills") is not None:
                lines.append(f"     Refills: {m['refills']}")
            leg = m.get("legibility", "clear")
            if leg != "clear":
                lines.append(f"     [legibility: {leg}]")
    if data.get("diagnoses"):
        lines.append("\nDIAGNOSES:")
        for d in data["diagnoses"]:
            lines.append(f"  - {d}")
    lines.append(f"\nIMAGE QUALITY: {data.get('image_quality', 'unknown')}")
    if data.get("notes"):
        lines.append(f"NOTES: {data['notes']}")
    return "\n".join(lines)


def _merge_typed(results: list[dict]) -> dict:
    if not results:
        return {}
    merged = dict(results[0])
    seen_dx: set[str] = set(merged.get("diagnoses") or [])
    for r in results[1:]:
        merged["tests"] = (merged.get("tests") or []) + (r.get("tests") or [])
        for d in r.get("diagnoses") or []:
            if d not in seen_dx:
                merged.setdefault("diagnoses", []).append(d)
                seen_dx.add(d)
        merged["warnings"] = (merged.get("warnings") or []) + (r.get("warnings") or [])
        for field in ("document_type", "report_date", "ordering_doctor",
                      "clinic_or_hospital", "summary_impression"):
            if not merged.get(field) and r.get(field):
                merged[field] = r[field]
    return merged


def process_document(source: RawSource, content: bytes, mime: str, kind: str) -> None:
    """
    Process a document and write OCR results onto the source row.
    Caller is responsible for db.commit(). Never raises.
    """
    max_bytes = settings.MAX_PDF_MB * 1024 * 1024
    if len(content) > max_bytes:
        source.ocr_status = "error"
        source.extraction_meta = {
            "error": (
                f"File size {len(content) // (1024 * 1024)} MB "
                f"exceeds {settings.MAX_PDF_MB} MB limit"
            ),
            "warnings": [],
        }
        return

    try:
        if kind == "pdf_typed":
            _process_pdf(source, content)
        else:
            _process_image(source, content, mime)
    except Exception as exc:
        logger.error("process_document failed for source %s: %s", source.id, exc)
        source.ocr_status = "error"
        source.extraction_meta = {"error": str(exc), "warnings": []}


def _process_pdf(source: RawSource, content: bytes) -> None:
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            page_texts = [p.extract_text() or "" for p in pdf.pages]
            page_count = len(pdf.pages)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc

    if page_count == 0:
        source.ocr_status = "unreadable"
        source.extraction_meta = {"error": "PDF has no pages", "warnings": [], "page_count": 0}
        return

    total_chars = sum(len(t) for t in page_texts)
    is_text_pdf = (total_chars / page_count) >= _CHARS_PER_PAGE_THRESHOLD

    if is_text_pdf:
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
        try:
            data = gemini_client.json_completion(_ocr_typed_prompt, full_text)
        except GeminiParseError as exc:
            source.ocr_status = "error"
            source.extraction_meta = {
                "error": str(exc), "warnings": [], "page_count": page_count,
                "model": "gemini-2.5-flash",
            }
            return
    else:
        try:
            pil_pages = convert_from_bytes(content, dpi=200)
        except Exception as exc:
            raise RuntimeError(f"Failed to rasterize PDF: {exc}") from exc

        page_count = len(pil_pages)
        page_results: list[dict] = []
        page_warnings: list[str] = []

        for i, pil_page in enumerate(pil_pages):
            img_bytes = _pil_to_jpeg(pil_page)
            try:
                result = gemini_client.vision_json_completion(
                    _ocr_typed_prompt,
                    [img_bytes],
                    "image/jpeg",
                    user_text=f"This is page {i + 1} of {page_count}.",
                )
                page_results.append(result)
            except GeminiParseError as exc:
                msg = f"Page {i + 1} OCR failed: {exc}"
                page_warnings.append(msg)
                logger.warning("Vision OCR source %s %s", source.id, msg)

        if not page_results:
            source.ocr_status = "error"
            source.extraction_meta = {
                "error": "All pages failed OCR",
                "warnings": page_warnings,
                "page_count": page_count,
                "model": "gemini-2.5-flash",
            }
            return

        data = _merge_typed(page_results)
        data.setdefault("warnings", [])
        data["warnings"].extend(page_warnings)

    extracted = _flatten_typed(data)
    if not extracted.strip():
        source.ocr_status = "unreadable"
        source.extraction_meta = {
            "warnings": (data.get("warnings") or []) + ["No content could be extracted"],
            "page_count": page_count,
            "model": "gemini-2.5-flash",
        }
        return

    source.ocr_status = "done"
    source.extracted_text = extracted
    source.extraction_meta = {**data, "page_count": page_count, "model": "gemini-2.5-flash"}


def _process_image(source: RawSource, content: bytes, mime: str) -> None:
    try:
        img_bytes, img_mime = _preprocess_image(content)
    except Exception as exc:
        logger.warning("Preprocessing failed for source %s, using original: %s", source.id, exc)
        img_bytes, img_mime = content, mime

    try:
        data = gemini_client.vision_json_completion(
            _ocr_prescription_prompt, [img_bytes], img_mime
        )
    except GeminiParseError as exc:
        source.ocr_status = "error"
        source.extraction_meta = {"error": str(exc), "warnings": [], "model": "gemini-2.5-flash"}
        return

    quality = data.get("image_quality", "fair")
    if quality == "unusable":
        source.ocr_status = "unreadable"
        source.extraction_meta = {
            "image_quality": "unusable",
            "warnings": (data.get("warnings") or [])
            + ["Image quality too poor to extract reliable content — do not use as fact"],
            "model": "gemini-2.5-flash",
        }
        return

    source.ocr_status = "done"
    source.extracted_text = _flatten_prescription(data)
    source.extraction_meta = {**data, "model": "gemini-2.5-flash"}
