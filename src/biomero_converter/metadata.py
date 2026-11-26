"""Read the metadata of a TIFF file, and the pixel size only a TIFF itself gives.

Every private tag, vendor XML and Exif field is collected into one plain
metadata dict, for the imaging metadata converter
(https://github.com/NL-BioImaging/imaging-metadata-converter) to map: it
also gives the pixel size, position and rotation the metadata states
(spatial_metadata), shared with the other readers. What is left here is
what only the TIFF holds: ImageJ's pixel size and the baseline resolution.
"""

from enum import Enum
import logging
from imaging_metadata_converter.units import normalise
from tifffile import TiffFile, xml2dict
from typing import Any, Dict, Optional
from xml.etree.ElementTree import ParseError


logger = logging.getLogger(__name__)

# TIFF reserves tag codes at or above this for a vendor's private use:
# https://www.awaresystems.be/imaging/tiff/tifftags/private.html
PRIVATE_TAG_CODE = 32768

# the Exif tag points at an IFD of standard acquisition fields, such as
# the exposure time, rather than at a vendor's own structure, so those
# fields are collected beside the other metadata instead of below it
EXIF_TAG_NAME = 'ExifTag'

# baseline tags naming the instrument that produced the image, which
# vendors that do not use a private tag at all still fill in
IDENTITY_TAG_NAMES = ('Make', 'Model', 'Software', 'HostComputer')

# ElementTree expands a namespaced xml attribute into a {namespace}name
# key. Attributes in this namespace, such as xsi:type and
# xsi:noNamespaceSchemaLocation, describe the document rather than the
# image, and vendors sprinkle them at every level of their schema.
XSI_NAMESPACE = '{http://www.w3.org/2001/XMLSchema-instance}'



def get_extra_metadata(metadata: dict) -> Dict[str, Any]:
    """Return the vendor metadata.
    """
    extra_metadata = {}
    for key, value in metadata.items():
        if key == EXIF_TAG_NAME and isinstance(value, dict):
            for name, field in value.items():
                extra_metadata.setdefault(name, field)
        else:
            # key by tag name, so that different vendor tags in one
            # file are kept side by side
            extra_metadata.setdefault(key, value)
    return repair_text(extra_metadata)


def get_extra_metadata_tiff(tif: TiffFile) -> Dict[str, Any]:
    """Return the vendor metadata in a TIFF file.

    Rather than reading the tags of particular vendors, every private tag
    is collected and normalised the same way, so that vendors which are
    not known here are picked up as well. TIFF reserves tag codes at or
    above 32768 for a vendor's own use, which is where instrument
    metadata ends up, whereas the baseline tags below that hold the
    bookkeeping needed to decode the pixels.

    The Exif IFD is the exception: its fields are standard rather than a
    vendor's own, so they are merged in beside the rest instead of being
    nested behind the name of the tag that points at them.
    """
    if tif.is_ome and tif.ome_metadata:
        return repair_text(decode_metadata(tif.ome_metadata))

    extra_metadata = {}
    # setdefault lets the first page win, as later pages tend to be
    # thumbnails or reduced resolutions; they only fill in missing fields
    for page in tif.pages:
        for tag in page.tags.values():
            if tag.code >= PRIVATE_TAG_CODE:
                value = decode_metadata(tag.value)
                if value in (None, '', {}):
                    continue
                if tag.name == EXIF_TAG_NAME and isinstance(value, dict):
                    for name, field in value.items():
                        extra_metadata.setdefault(name, field)
                else:
                    # key by tag name, so that different vendor tags in one
                    # file are kept side by side
                    extra_metadata.setdefault(tag.name, value)
            elif tag.name in IDENTITY_TAG_NAMES:
                extra_metadata.setdefault(tag.name, tag.value)
    return repair_text(extra_metadata)


def decode_metadata(value: Any) -> Any:
    """Decode a metadata value into the structure it holds.

    Vendors store their metadata as an xml document or as a nested
    mapping, so parse the xml into a mapping, keeping its root element,
    such as OME, FeiImage or Fibics, which names the vendor's schema.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, str):
        parsed = parse_xml(value)
        if parsed is None:
            return value
        value = parsed
    return drop_document_details(value)


def parse_xml(value: str) -> Optional[Dict]:
    """Return value parsed as an xml document, or None if it is not one.

    The xml declaration is optional, and vendors do leave it out, so
    rather than looking for one, hand anything that opens like a
    document to the parser and let it decide.
    """
    if not value.lstrip().startswith('<'):
        return None
    try:
        return xml2dict(value)
    except ParseError:
        return None


def drop_document_details(value: Any) -> Any:
    """Recursively drop the entries describing the xml document itself.

    The plumbing appears at every level of a vendor's schema, not just
    at the top, so this has to walk the whole tree.
    """
    if isinstance(value, dict):
        return {key: drop_document_details(item)
                for key, item in value.items()
                if not is_document_detail(key, item)}
    if isinstance(value, list):
        return [drop_document_details(item) for item in value]
    return value


def is_document_detail(key: Any, value: Any) -> bool:
    """Return whether an entry describes the document, not the image."""
    return ((isinstance(key, str) and key.startswith(XSI_NAMESPACE))
            or (isinstance(value, str) and '.xsd' in value.lower()))


def repair_text(value: Any) -> Any:
    """Recursively put back the characters a broken encoding garbled.

    Only the sequences known to be garbled are replaced, rather than
    guessing at the encoding of every string, which would mangle text
    that was read correctly.
    """
    if isinstance(value, str):
        for garbled, meant in MOJIBAKE.items():
            value = value.replace(garbled, meant)
        return value
    if isinstance(value, dict):
        return {key: repair_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_text(item) for item in value]
    return value


# text a vendor wrote in one encoding and tifffile read back as latin-1,
# keyed by how it reads, with the character that was meant. Helios writes
# the micro sign as the Greek mu in GBK, whose two bytes read as '¦Ì'.
MOJIBAKE = {
    '¦Ì': 'µ',
}

# a pixel below this is smaller than an atom, and one above it is wider
# than a hand, so a value outside the range is a misread rather than a
# measurement, whatever the metadata claims
PLAUSIBLE_PIXEL_SIZE_UM = (1e-5, 1e4)

# resolutions every second consumer writer fills in, in pixels per inch,
# which say what a viewer should print at rather than what was imaged
SCREEN_RESOLUTIONS = (72, 96)


def get_pixel_size_um(tif: TiffFile, stated: Dict[str, float]) -> Dict[str, float]:
    """Return the size of a pixel in micrometres, per axis.

    The sources are tried in the order they can be trusted: what ImageJ
    writes, then what the metadata states (mapped by the imaging metadata
    converter: `stated`), and last the baseline TIFF resolution, which is
    the one every writer fills in whether it knows the answer or not.
    """
    pixel_size = {}
    if tif.is_imagej and tif.imagej_metadata:
        pixel_size.update(imagej_pixel_size(tif))
    for axis, size in list(stated.items()) + list(resolution_pixel_size(tif).items()):
        pixel_size.setdefault(axis, size)
    return {axis: size for axis, size in pixel_size.items()
            if is_plausible_pixel_size(size)}


def is_plausible_pixel_size(size: Any) -> bool:
    """Return whether a pixel size can be a measurement at all."""
    low, high = PLAUSIBLE_PIXEL_SIZE_UM
    return isinstance(size, float) and low <= size <= high


def resolution_pixel_size(tif: TiffFile) -> Dict:
    """Return the pixel size from the baseline TIFF resolution.

    The resolution is a rational, and a writer with nothing to say
    fills it in anyway, so read it only where it can mean something.
    """
    pixel_size = {}
    tags = tif.pages.first.tags
    unit = getattr(tags.valueof('ResolutionUnit'), 'name', '')
    for axis, name in (('x', 'XResolution'), ('y', 'YResolution')):
        value = tags.valueof(name)
        if isinstance(value, tuple) and all(value):
            numerator, denominator = value
            if numerator == denominator:
                # the rational cancels out, so the ratio says nothing and
                # only the numerator is left to read. A resolution that
                # large is pixels per metre, whatever the unit tag claims,
                # which is filled in by default as readily as the value.
                size = normalise(1 / numerator, 'm', 'µm')
            elif unit == 'CENTIMETER':
                size = normalise(denominator / numerator, 'cm', 'µm')
            elif unit == 'INCH' and numerator / denominator not in SCREEN_RESOLUTIONS:
                size = normalise(denominator / numerator, 'in', 'µm')
            else:
                # no unit at all, or one of the resolutions a writer fills
                # in to say how large to print the image on a screen
                size = None
            if size is not None:
                pixel_size[axis] = size
    return pixel_size


def imagej_pixel_size(tif: TiffFile) -> Dict:
    """Return the pixel size ImageJ writes, which names its own unit."""
    metadata = tif.imagej_metadata
    unit = metadata.get('unit', '')
    pixel_size = {}
    for axis, name in (('x', 'XResolution'), ('y', 'YResolution')):
        value = tif.pages.first.tags.valueof(name)
        size = normalise(value[1] / value[0], unit, 'µm') if isinstance(value, tuple) and all(value) else None
        if size is not None:
            pixel_size[axis] = size
    spacing = normalise(metadata['spacing'], unit, 'µm') if 'spacing' in metadata else None
    if spacing is not None:
        pixel_size['z'] = spacing
    return pixel_size
