"""Cross-checks OCR-extracted fields against MRZ/barcode-extracted fields, flagging mismatches and 'unverifiable' documents."""

from dataclasses import dataclass

from document.mrz_parser import MrzResult

# Rough heuristic threshold for the barcode/OCR word-overlap check below: not calibrated against
# real captured pairs, same caveat as config.settings.DEFAULT_FACE_MATCH_THRESHOLD.
_BARCODE_OVERLAP_VALID_RATIO = 0.5


@dataclass
class FieldComparison:
    """The result of comparing one field as read from a machine-readable source (MRZ) against the OCR text."""

    field_name: str
    machine_read_value: str
    found_in_ocr: bool


@dataclass
class CrosscheckResult:
    """The overall document validation outcome: status, the field-level comparisons behind it, and any explanatory notes."""

    validation_status: str  # 'valid' / 'review' / 'invalid' / 'unverifiable'
    field_comparisons: list[FieldComparison]
    notes: list[str]


def _field_appears_in_ocr(value: str, ocr_text: str) -> bool:
    """Check whether a machine-read field value shows up (case-insensitively, MRZ '<' fill chars stripped) somewhere in the raw OCR text."""
    cleaned_value = value.replace("<", " ").strip()
    if not cleaned_value:
        return True  # an unused/empty MRZ field has nothing to cross-check, so it can't fail the check
    return cleaned_value.upper() in ocr_text.upper()


def crosscheck_mrz_against_ocr(mrz_result: MrzResult, ocr_text: str) -> CrosscheckResult:
    """Cross-check every parsed MRZ field against the raw OCR text and decide the resulting validation status.

    The MRZ's own checksum digits are already validated by mrz_parser.py; this function
    additionally confirms the OCR pass over the visible printed fields agrees with what the MRZ
    encodes, since a forged or altered document is more likely to have the visible text and the
    MRZ disagree than to have both independently and consistently forged. A checksum failure is
    treated as 'invalid' outright (the MRZ itself is structurally wrong, no point cross-checking
    fields from it); field-level disagreement against otherwise-valid MRZ data is treated as
    'review', not an automatic reject, since it may just as easily be an OCR misread.
    """
    if not mrz_result.checksum_valid:
        return CrosscheckResult(
            validation_status="invalid",
            field_comparisons=[],
            notes=[f"MRZ checksum validation failed: {', '.join(mrz_result.failed_checks)}"],
        )

    fields_to_check = ["surname", "name", "document_number", "birth_date", "expiry_date"]
    comparisons = [
        FieldComparison(
            field_name=field_name,
            machine_read_value=mrz_result.fields[field_name],
            found_in_ocr=_field_appears_in_ocr(mrz_result.fields[field_name], ocr_text),
        )
        for field_name in fields_to_check
        if mrz_result.fields.get(field_name)
    ]

    mismatches = [comparison for comparison in comparisons if not comparison.found_in_ocr]
    if mismatches:
        return CrosscheckResult(
            validation_status="review",
            field_comparisons=comparisons,
            notes=[f"{comparison.field_name} from MRZ not found in OCR text" for comparison in mismatches],
        )
    return CrosscheckResult(validation_status="valid", field_comparisons=comparisons, notes=[])


def crosscheck_barcode_against_ocr(barcode_texts: list[str], ocr_text: str) -> CrosscheckResult:
    """Cross-check decoded PDF417 barcode text against the raw OCR text.

    Unlike MRZ fields, this app does not parse individual named fields out of the barcode payload
    (see barcode_parser.py -- the Swedish driving licence PDF417 field layout is unvalidated), so
    this can only check whether the barcode decoded at all and whether its raw text substantially
    overlaps with the OCR text, rather than comparing named fields one by one. This is a
    deliberately weaker signal than the MRZ path and should not be presented to the operator as
    equivalent confidence.
    """
    if not barcode_texts:
        return CrosscheckResult(validation_status="unverifiable", field_comparisons=[], notes=["No barcode decoded"])

    combined_barcode_text = " ".join(barcode_texts).upper()
    ocr_words = [word for word in ocr_text.upper().split() if len(word) > 2]

    if not ocr_words:
        overlap_ratio = 0.0
    else:
        matching_word_count = sum(1 for word in ocr_words if word in combined_barcode_text)
        overlap_ratio = matching_word_count / len(ocr_words)

    status = "valid" if overlap_ratio >= _BARCODE_OVERLAP_VALID_RATIO else "review"
    notes = [
        f"OCR/barcode word overlap ratio: {overlap_ratio:.2f} "
        "(field-level barcode parsing not implemented -- see README Known limitations)"
    ]
    return CrosscheckResult(validation_status=status, field_comparisons=[], notes=notes)


def determine_validation_status(mrz_result: MrzResult, barcode_texts: list[str], ocr_text: str) -> CrosscheckResult:
    """Decide the overall validation_status for a scanned document: cross-check via MRZ if present, else via barcode, else mark it unverifiable.

    A document with neither an MRZ nor a barcode has only OCR text to go on, with no independent
    machine-readable source to cross-check it against. This is explicitly marked 'unverifiable'
    rather than 'valid' or any other status implying a passed check, so OCR-only documents are
    never visually indistinguishable from cross-validated ones in the UI.
    """
    if mrz_result.found:
        return crosscheck_mrz_against_ocr(mrz_result, ocr_text)
    if barcode_texts:
        return crosscheck_barcode_against_ocr(barcode_texts, ocr_text)
    return CrosscheckResult(
        validation_status="unverifiable",
        field_comparisons=[],
        notes=["No MRZ or barcode found on document; only OCR text available"],
    )
