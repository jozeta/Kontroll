"""Tests for the pure document logic: printed-date extraction, MRZ-style dates and ages, AAMVA
barcode parsing, and the under-18/expiry flags. These need only the standard library and the small
`mrz` package (no camera, Qt, OpenCV or InsightFace), so they run anywhere, CI included.

    python -m pytest -q
"""

from datetime import date, timedelta

import pytest

from document.aamva_parser import parse_aamva_barcode, parse_first_aamva_barcode
from document.dates import calculate_age, extract_date_from_line, parse_mrz_style_date
from document.field_crosscheck import MINOR_AGE_THRESHOLD_YEARS, check_person_flags


# --- printed dates (OCR lines) ----------------------------------------------------------------
@pytest.mark.parametrize(
    "line, expected",
    [
        ("4b. 2031-05-17", "2031-05-17"),  # Swedish ISO style
        ("3. 28.07.1981 Stockholm", "1981-07-28"),  # EU dd.mm.yyyy
        ("28.07 1981", "1981-07-28"),  # OCR turned a '.' into a space
        ("18121972", "1972-12-18"),  # OCR dropped both separators
        ("2207 1968", "1968-07-22"),  # ...or only one
    ],
)
def test_extract_date_from_line_handles_ocr_mangled_separators(line, expected):
    assert extract_date_from_line(line) == expected


@pytest.mark.parametrize("line", ["no date here", "32.01.1990", "01.13.1990", "01.01.1850", ""])
def test_extract_date_from_line_rejects_impossible_dates(line):
    assert extract_date_from_line(line) is None


# --- MRZ-style dates and age --------------------------------------------------------------------
def test_mrz_birth_date_century_never_lies_in_the_future():
    this_year_2 = date.today().year % 100
    future_2 = (this_year_2 + 1) % 100
    born = parse_mrz_style_date(f"{future_2:02d}0101", assume_past=True)
    assert born is not None and born <= date.today()


def test_mrz_expiry_dates_are_20xx_and_iso_dates_pass_through():
    assert parse_mrz_style_date("310517", assume_past=False) == date(2031, 5, 17)
    assert parse_mrz_style_date("1981-07-28", assume_past=True) == date(1981, 7, 28)


@pytest.mark.parametrize("raw", ["", "12345", "ABCDEF", "991945", "1234567"])
def test_malformed_mrz_dates_are_unknown_not_a_crash(raw):
    assert parse_mrz_style_date(raw, assume_past=True) is None


def test_calculate_age_counts_whole_years_around_the_birthday():
    born = date(2000, 6, 15)
    assert calculate_age(born, as_of=date(2018, 6, 14)) == 17
    assert calculate_age(born, as_of=date(2018, 6, 15)) == 18


# --- AAMVA (US/Canadian licence barcodes) -------------------------------------------------------
SAMPLE = (
    "@\n\x1e\rANSI 636014090102DL00410288ZC03290034DLDAQD1234567\n"
    "DCSDOE\nDACJOHN\nDADNONE\nDBB01151990\nDBA06302030\nDCGUSA\nDAJCA\n"
)


def test_aamva_fields_are_extracted_with_the_shared_key_names():
    r = parse_aamva_barcode(SAMPLE)
    assert r.found and not r.is_id_card
    assert r.fields == {
        "surname": "DOE",
        "name": "JOHN",  # middle name "NONE" is dropped
        "birth_date": "1990-01-15",
        "expiry_date": "2030-06-30",
        "document_number": "D1234567",
        "country": "USA",
        "jurisdiction": "CA",
    }


def test_aamva_id_card_subfile_and_bad_dates():
    r = parse_aamva_barcode(SAMPLE.replace("0102DL", "0102ID").replace("DBB01151990", "DBB13451990"))
    assert r.is_id_card
    assert "birth_date" not in r.fields  # month 13 is not a date


def test_non_aamva_payloads_are_not_found():
    assert not parse_aamva_barcode("just some PDF417 text").found
    assert not parse_first_aamva_barcode(["x", "y"]).found
    assert parse_first_aamva_barcode(["x", SAMPLE]).fields["surname"] == "DOE"


# --- under-18 and expiry flags ------------------------------------------------------------------
def test_minor_and_expired_flags():
    today = date.today()
    minor_birth = date(today.year - (MINOR_AGE_THRESHOLD_YEARS - 1), 1, 1).isoformat()
    expired = (today - timedelta(days=1)).isoformat()
    r = check_person_flags({"birth_date": minor_birth, "expiry_date": expired})
    assert r.is_minor and r.is_expired
    assert len(r.notes) == 2


def test_adult_with_valid_document_has_no_flags():
    today = date.today()
    r = check_person_flags({"birth_date": "1980-01-01", "expiry_date": date(today.year + 2, 1, 1).isoformat()})
    assert not r.is_minor and not r.is_expired and r.notes == []


def test_missing_dates_give_no_flags_rather_than_a_guess():
    r = check_person_flags({})
    assert (r.is_minor, r.is_expired, r.age_years, r.expiry_date_display) == (False, False, None, None)
