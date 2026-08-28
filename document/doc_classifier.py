"""Classifies document type (passport / national ID / driving licence) and issuing country."""

from dataclasses import dataclass

from document.aamva_parser import parse_first_aamva_barcode
from document.dl_field_parser import parse_driving_licence_fields
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


def _classify_from_ocr_keywords(ocr_text: str) -> DocumentClassification | None:
    """Classify a driving licence or Swedish national ID card from OCR keywords alone, or return None if none matched.

    Deliberately independent of whether a barcode was actually decoded: the printed title text is
    readable via OCR even when the PDF417 barcode itself fails to decode (glare, angle,
    resolution) -- see document/dl_field_parser.py, which extracts a driving licence's name/
    birth-date fields the same OCR-only way, independent of barcode decode success, using the
    EU-harmonized field-number layout that's already language-independent. A US driver's licence/
    ID is instead classified from its AAMVA barcode payload when that decodes successfully (see
    _classify_from_aamva_barcode above) -- this OCR-keyword path is only its fallback for when
    that decode fails too.

    'DRIVING LICENCE'/'DRIVING LICENSE'/'DRIVER LICENSE' plus each EU member state's own-language
    title (Council Directive 2006/126/EC, Annex I mandates the national-language title as the
    main heading) are all matched, factual translations rather than guessed layouts -- this is
    just naming detection, independent of document/dl_field_parser.py's field-number-anchored
    extraction, which already works the same regardless of which country's licence it is. Per-
    keyword country codes are a best guess from which title matched, not a validated per-country
    layout the way the Swedish path has been diagnosed against real captures.

    Deliberately does NOT match on a bare country name alone (a prior version of this check did,
    and was found misclassifying Swedish national ID cards -- which also print 'SVERIGE'/'SWEDEN'
    -- as drivers_licence whenever their MRZ wasn't found/read, e.g. a front-only capture). A
    country name alone says only "this is a document from that country", not which kind.

    Also matches 'KORKORT' (no diaeresis) alongside 'KÖRKORT': real captures have shown Tesseract
    dropping the 'Ö''s dots and reading it as a bare 'O', which is a much more common failure mode
    than dropping the whole word.
    """
    normalized_text = ocr_text.upper()
    # (keyword, doc_country) -- doc_country is set purely from which title matched, so a licence
    # printing more than one of these (e.g. an EU-format licence's English subtitle alongside its
    # national-language title) resolves to whichever is checked first below.
    drivers_licence_keywords: tuple[tuple[str, str | None], ...] = (
        ("KÖRKORT", "SWE"), ("KORKORT", "SWE"),
        ("FÜHRERSCHEIN", "DEU"), ("FUHRERSCHEIN", "DEU"),
        ("PERMIS DE CONDUIRE", "FRA"),
        ("PERMISO DE CONDUCIR", "ESP"), ("PERMISO DE CONDUCCIÓN", "ESP"),
        ("PATENTE DI GUIDA", "ITA"),
        ("RIJBEWIJS", "NLD"),
        ("CARTA DE CONDUÇÃO", "PRT"),
        ("PRAWO JAZDY", "POL"),
        ("KØREKORT", "DNK"),
        ("FØRERKORT", "NOR"),
        ("AJOKORTTI", "FIN"),
        ("DRIVER LICENSE", "USA"), ("DRIVER'S LICENSE", "USA"),
        ("DRIVING LICENCE", None), ("DRIVING LICENSE", None),
    )
    for keyword, doc_country in drivers_licence_keywords:
        if keyword in normalized_text:
            return DocumentClassification(doc_type="drivers_licence", doc_country=doc_country, classified_from="ocr_keywords")

    # 'IDENTITETSKORT' (from 'NATIONELLT IDENTITETSKORT') has been reliably OCR'd intact across
    # every real Swedish national ID card capture seen so far -- unlike 'IDENTITY CARD' printed
    # right below it, which has come out as noise ('IDENTITY CARO', 'TOENTITY CARD') often enough
    # not to rely on alone. Only reached for a front-only capture (no MRZ found) -- see
    # document/national_id_parser.py for the OCR-only field extraction that follows from this.
    if "IDENTITETSKORT" in normalized_text:
        return DocumentClassification(doc_type="national_id", doc_country="SWE", classified_from="ocr_keywords")

    # 'PASSPORT'/'PASSEPORT' are the ICAO-mandated bilingual (English/French) heading printed on
    # every passport regardless of issuing country, so unlike the Sweden-specific keywords above,
    # their presence alone doesn't imply a Swedish passport -- only set doc_country when a bare
    # country name is also present (see _country_from_bare_name). Only reached once MRZ reading
    # has already failed (e.g. a heavily noisy capture -- see document/passport_parser.py for the
    # OCR-only fallback that follows from this).
    if "PASSPORT" in normalized_text or "PASSEPORT" in normalized_text:
        return DocumentClassification(
            doc_type="passport", doc_country=_country_from_bare_name(normalized_text), classified_from="ocr_keywords"
        )

    return None


def _country_from_bare_name(normalized_text: str) -> str | None:
    """Best-effort issuing-country guess from a bare country name in the OCR text, independent of any document-type-specific keyword.

    Used when a document-type match didn't itself carry an unambiguous country -- the ICAO-
    mandated 'PASSPORT'/'PASSEPORT' heading above says nothing about which country issued it, and
    neither does dl_field_structure below matching (the EU numbered-field layout is shared by
    every member state). Only Sweden is checked -- the only country this has actually been
    diagnosed as needed for so far, via two different real captures: a passport OCR pass that
    found 'PASSPORT' but needed 'SVERIGE'/'SWEDEN' too before assuming Swedish, and a driving
    licence whose title OCR'd as 'FEETIK�RKORT' (unrecognizable) while 'SVERIGE', printed
    separately elsewhere on the same card, still came through cleanly right next to it. Extend
    with more countries' names only once a similar real gap turns up for one of them.
    """
    if "SVERIGE" in normalized_text or "SWEDEN" in normalized_text:
        return "SWE"
    return None


def _classify_from_dl_field_structure(ocr_text: str) -> DocumentClassification | None:
    """Classify as a driving licence purely from its EU-harmonized numbered-field layout (fields 1/2/3/4b -- see document/dl_field_parser.py) actually parsing, independent of whether any title keyword above was itself readable.

    Last-resort fallback for a driving licence whose printed title ('KÖRKORT'/'DRIVING LICENCE'/
    ...) OCR'd too badly for _classify_from_ocr_keywords to recognize at all -- diagnosed against
    a real capture where 'KÖRKORT' came out as 'FEETIK�RKORT', matching neither the diaeresis-
    dropped 'KORKORT' fallback nor the full keyword, even though the numbered fields themselves
    (Council-Directive-mandated plain digits, not printed words in any particular language) still
    read and parsed correctly. Reusing dl_field_parser.py's own success/failure as the signal ties
    classification directly to actual extractability instead of duplicating a second, separate
    "does this look like a driving licence" heuristic. Only reached once every title-keyword check
    above has already failed to match anything at all, so this doesn't get a chance to compete
    with (or override) a document that title-matched cleanly as a national ID card or passport.

    doc_country still comes from _country_from_bare_name rather than being left None here: the
    same real capture that motivated this whole fallback ('KÖRKORT' unreadable) had 'SVERIGE'
    printed separately elsewhere on the card and read just fine -- the title being unreadable
    doesn't mean every other printed country reference on the card is too.

    Requires personal_number plus a name field specifically, not just parse_driving_licence_fields'
    own found flag (any single non-empty field) -- diagnosed against a real capture of an entirely
    different document (a passport, photographed as a two-page open-book spread rather than a
    single page, which this app's pipeline isn't designed to read cleanly) whose badly garbled OCR
    text happened to contain exactly one coincidental field-shaped fragment ("surname: OPEZYY"),
    which was enough to misclassify it as a driving licence when any one field sufficed. The real
    KÖRKORT-unreadable capture that motivated this fallback read all five fields cleanly, so
    requiring personal_number (a precise 6-digit + separator + 4-digit shape, much less likely to
    appear by coincidence than a short uppercase fragment) plus a name keeps that case working
    while rejecting single-fragment false positives from unrelated garbled text.
    """
    fields = parse_driving_licence_fields(ocr_text).fields
    if "personal_number" in fields and ("surname" in fields or "name" in fields):
        return DocumentClassification(
            doc_type="drivers_licence", doc_country=_country_from_bare_name(ocr_text.upper()), classified_from="dl_field_structure"
        )
    return None


def _classify_from_aamva_barcode(barcode_texts: list[str]) -> DocumentClassification | None:
    """Classify a US/Canadian driver's license or ID card from its AAMVA-format PDF417 barcode payload, or return None if no decoded barcode parses as one.

    Checked ahead of the OCR-keyword fallback below: an AAMVA payload is a deterministic,
    precisely specified match (see document/aamva_parser.py) rather than a heuristic guess from
    printed text, so it's the stronger signal whenever both happen to be available.
    """
    aamva_result = parse_first_aamva_barcode(barcode_texts)
    if not aamva_result.found:
        return None
    doc_type = "national_id" if aamva_result.is_id_card else "drivers_licence"
    return DocumentClassification(
        doc_type=doc_type, doc_country=aamva_result.fields.get("country"), classified_from="aamva_barcode"
    )


def classify_document(mrz_result: MrzResult, barcode_texts: list[str], ocr_text: str) -> DocumentClassification:
    """Determine document type and issuing country from the strongest available evidence: MRZ first, then an AAMVA barcode, then OCR keywords, then the EU driving-licence numbered-field structure, then bare barcode presence, then unknown."""
    if mrz_result.found:
        return _classify_from_mrz(mrz_result)

    aamva_classification = _classify_from_aamva_barcode(barcode_texts)
    if aamva_classification is not None:
        return aamva_classification

    ocr_classification = _classify_from_ocr_keywords(ocr_text)
    if ocr_classification is not None:
        return ocr_classification

    dl_structure_classification = _classify_from_dl_field_structure(ocr_text)
    if dl_structure_classification is not None:
        return dl_structure_classification

    if barcode_texts:
        return DocumentClassification(doc_type="drivers_licence", doc_country=None, classified_from="barcode_presence")

    return DocumentClassification(doc_type="unknown", doc_country=None, classified_from="unknown")
