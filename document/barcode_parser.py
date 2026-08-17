"""Decodes PDF417 barcodes found on the document image (via zxing-cpp)."""

import numpy as np
import zxingcpp


def decode_pdf417_barcodes(image: np.ndarray) -> list[str]:
    """Scan a document image for PDF417 barcodes and return the decoded text of each one found.

    Restricted to the PDF417 format specifically, rather than zxing-cpp's default of scanning for
    every supported symbology, since that is the only barcode type this app's target documents
    (e.g. Swedish driving licences) are expected to carry -- narrowing the search space also cuts
    down false-positive decodes from incidental print patterns elsewhere on the document.

    This returns raw decoded text only, not parsed fields: the internal field layout of a Swedish
    driving licence's PDF417 payload has not been validated against a real sample and must not be
    assumed to match AAMVA (US) formats -- see README "Known limitations". Structured field
    extraction from barcode payloads is therefore out of scope until that validation happens.
    """
    barcodes = zxingcpp.read_barcodes(image, formats=zxingcpp.PDF417)
    return [barcode.text for barcode in barcodes if barcode.valid]
