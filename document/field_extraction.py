"""Runs the document classification + structured field extraction portion of the capture pipeline
against a single document frame, independent of face matching.

Extracted out of ui/main_window.py::_analyze_capture so the exact same logic can be reused by
ui/image_viewer.py::ImageEditDialog's "what would this image extract to" preview -- the two must
never independently reimplement (and risk drifting apart on) which parser wins for which document
type; see _analyze_capture's own docstring for why it takes plain frames rather than a ScanRecord.
"""

from dataclasses import dataclass, field

import numpy as np

from document.aamva_parser import parse_first_aamva_barcode
from document.barcode_parser import decode_pdf417_barcodes
from document.card_detector import detect_and_crop_card
from document.doc_classifier import DocumentClassification, classify_document
from document.dl_field_parser import parse_driving_licence_fields
from document.eu_id_card_parser import parse_eu_id_card_fields
from document.mrz_locator import locate_mrz_band
from document.mrz_parser import MrzResult, parse_and_validate_mrz
from document.national_id_parser import parse_national_id_fields
from document.ocr import OcrWord, correct_text_orientation, extract_mrz_candidate_text
from document.passport_parser import parse_passport_fields


@dataclass
class DocumentFieldsResult:
    """Every derived signal document/field_extraction.py::extract_document_fields produces from one document frame."""

    doc_analysis_frame: np.ndarray  # cropped-to-card, upright-oriented -- see detect_and_crop_card/correct_text_orientation
    ocr_words: list[OcrWord]
    ocr_text: str
    mrz_result: MrzResult
    barcode_texts: list[str]
    doc_classification: DocumentClassification
    doc_fields: dict[str, str] = field(default_factory=dict)


def extract_document_fields(doc_frame: np.ndarray) -> DocumentFieldsResult:
    """Crop to the card, OCR it, locate/validate an MRZ or PDF417 barcode if present, classify the document type/country, and extract whichever structured fields (name, birth date, personal number, ...) that classification's parser recovers.

    See ui/main_window.py::_analyze_capture's own comments (this function's original home) for the
    detailed rationale behind each step and parser choice -- unchanged here, just extracted so a
    second caller (ImageEditDialog's field-extraction preview) can reuse it verbatim.
    """
    doc_analysis_frame = detect_and_crop_card(doc_frame)
    doc_analysis_frame, ocr_words, ocr_text = correct_text_orientation(doc_analysis_frame)

    mrz_result = parse_and_validate_mrz(ocr_text)
    if not mrz_result.found:
        mrz_band = locate_mrz_band(doc_analysis_frame)
        mrz_candidate_text = extract_mrz_candidate_text(mrz_band)
        mrz_result = parse_and_validate_mrz(mrz_candidate_text)
    barcode_texts = decode_pdf417_barcodes(doc_analysis_frame)
    doc_classification = classify_document(mrz_result, barcode_texts, ocr_text)

    if mrz_result.found:
        doc_fields = mrz_result.fields
        if doc_classification.doc_type == "passport" and not (doc_fields.get("surname") and doc_fields.get("name")):
            passport_fields = parse_passport_fields(ocr_text).fields
            doc_fields = {**doc_fields, **{key: passport_fields[key] for key in ("surname", "name") if passport_fields.get(key)}}
    elif doc_classification.classified_from == "aamva_barcode":
        doc_fields = parse_first_aamva_barcode(barcode_texts).fields
    elif doc_classification.doc_type == "drivers_licence":
        dl_fields_result = parse_driving_licence_fields(ocr_text)
        doc_fields = dl_fields_result.fields if dl_fields_result.found else {}
    elif doc_classification.doc_type == "national_id":
        national_id_result = parse_national_id_fields(ocr_text)
        if national_id_result.found:
            doc_fields = national_id_result.fields
        else:
            eu_id_card_result = parse_eu_id_card_fields(ocr_text)
            doc_fields = eu_id_card_result.fields if eu_id_card_result.found else {}
            if doc_classification.doc_country is None and doc_fields.get("nationality"):
                doc_classification.doc_country = doc_fields["nationality"]
    elif doc_classification.doc_type == "passport":
        passport_result = parse_passport_fields(ocr_text)
        doc_fields = passport_result.fields if passport_result.found else {}
    else:
        doc_fields = {}

    return DocumentFieldsResult(
        doc_analysis_frame=doc_analysis_frame,
        ocr_words=ocr_words,
        ocr_text=ocr_text,
        mrz_result=mrz_result,
        barcode_texts=barcode_texts,
        doc_classification=doc_classification,
        doc_fields=doc_fields,
    )
