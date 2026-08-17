"""Classifies document type (passport / national ID / driving licence) and issuing country."""

from dataclasses import dataclass

from document.mrz_parser import MrzResult


@dataclass
class DocumentClassification:
    """The determined document type and issuing country, plus what evidence the determination was based on."""

    doc_type: str  # 'passport' / 'national_id' / 'drivers_licence' / 'unknown'
    doc_country: str | None  # ISO 3166-1 alpha-3 code where known, else None
    classified_from: str  # 'mrz_document_code' / 'ocr_keywords' / 'barcode_presence' / 'unknown'


def _classify_from_mrz(mrz_result: MrzResult) -> DocumentClassification:
    """Classify a document using its MRZ document-code and country fields, per ICAO 9303 document type codes.

    ICAO 9303 document type codes: a leading 'P' (e.g. 'P<') means passport. A leading 'I', 'A',
    or 'C' covers the various national ID card variants seen on TD1-format MRZs. Driving licences
    do not carry an MRZ under ICAO 9303, so this path is only reached once an MRZ has already been
    located and parsed.
    """
    document_type_code = mrz_result.fields.get("document_type", "").strip("<").upper()
    country_code = mrz_result.fields.get("country", "").strip("<").upper()

    if document_type_code.startswith("P"):
        doc_type = "passport"
    elif document_type_code.startswith(("I", "A", "C")):
        doc_type = "national_id"
    else:
        doc_type = "unknown"

    return DocumentClassification(
        doc_type=doc_type,
        doc_country=country_code or None,
        classified_from="mrz_document_code",
    )


def _classify_from_barcode_presence(ocr_text: str) -> DocumentClassification:
    """Classify a document that has a decoded PDF417 barcode but no MRZ.

    Only Swedish driving licences are supported end-to-end for barcode-based documents in v1, so
    this is a narrow keyword check against the OCR text rather than a general barcode-content
    classifier -- extending this to other countries' barcode-carrying documents is future work.
    """
    normalized_text = ocr_text.upper()
    swedish_keywords = ("KÖRKORT", "DRIVING LICENCE", "SWEDEN", "SVERIGE")
    if any(keyword in normalized_text for keyword in swedish_keywords):
        return DocumentClassification(doc_type="drivers_licence", doc_country="SWE", classified_from="ocr_keywords")
    return DocumentClassification(doc_type="drivers_licence", doc_country=None, classified_from="barcode_presence")


def classify_document(mrz_result: MrzResult, barcode_texts: list[str], ocr_text: str) -> DocumentClassification:
    """Determine document type and issuing country from the strongest available evidence: MRZ first, then barcode presence, then unknown."""
    if mrz_result.found:
        return _classify_from_mrz(mrz_result)
    if barcode_texts:
        return _classify_from_barcode_presence(ocr_text)
    return DocumentClassification(doc_type="unknown", doc_country=None, classified_from="unknown")
