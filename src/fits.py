'''Minimal reader for (gzipped) FITS binary tables, without astropy.

Supports the fixed-width column types used by the ASTRA outputs
(L, B, I, J, K, E, D with optional repeat counts). Variable-length arrays and
scaled columns (TSCAL/TZERO) are rejected explicitly instead of being misread.
'''
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

_BLOCK = 2880
_TYPES = {'L': '>i1', 'B': '>u1', 'I': '>i2', 'J': '>i4', 'K': '>i8', 'E': '>f4', 'D': '>f8'}


def _parse_value(text: str):
    text = text.strip()
    if text.startswith("'"):
        end = text.find("'", 1)
        while end != -1 and end + 1 < len(text) and text[end + 1] == "'":
            end = text.find("'", end + 2)
        return text[1:end].replace("''", "'").rstrip()
    value = text.split('/', 1)[0].strip()
    if value in ('T', 'F'):
        return value == 'T'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_header(raw: bytes, pos: int) -> tuple[dict, int]:
    header = {}
    while True:
        if pos + _BLOCK > len(raw):
            raise ValueError('Truncated FITS header.')
        block = raw[pos:pos + _BLOCK]
        pos += _BLOCK
        for i in range(36):
            card = block[80 * i:80 * (i + 1)].decode('ascii', errors='replace')
            key = card[:8].strip()
            if key == 'END':
                return header, pos
            if card[8:10] == '= ' and key not in header:
                header[key] = _parse_value(card[10:])


def read_fits_table(path: str | Path, extname: str | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    '''Return (header, columns) of the first BINTABLE (or the one named ``extname``).'''
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as handle:
        raw = handle.read()
    header, pos = _read_header(raw, 0)
    if header.get('SIMPLE') is not True:
        raise ValueError(f'Not a FITS file: {path}')
    pos += _data_size(header)
    while pos < len(raw):
        header, pos = _read_header(raw, pos)
        size = _data_size(header)
        if header.get('XTENSION') == 'BINTABLE' and (extname is None or header.get('EXTNAME') == extname):
            return header, _decode_table(header, raw[pos:pos + int(header['NAXIS1']) * int(header['NAXIS2'])], path)
        pos += size
    raise ValueError(f'No BINTABLE{f" named {extname}" if extname else ""} in {path}')


def _data_size(header: dict) -> int:
    naxis = int(header.get('NAXIS', 0))
    if naxis == 0:
        return 0
    n = abs(int(header.get('BITPIX', 8))) // 8
    for i in range(1, naxis + 1):
        n *= int(header[f'NAXIS{i}'])
    n = (n + int(header.get('PCOUNT', 0))) * int(header.get('GCOUNT', 1))
    return (n + _BLOCK - 1) // _BLOCK * _BLOCK


def _decode_table(header: dict, data: bytes, path: Path) -> dict[str, np.ndarray]:
    width, rows = int(header['NAXIS1']), int(header['NAXIS2'])
    if len(data) != width * rows:
        raise ValueError(f'Truncated FITS table data: {path}')
    fields = []
    for i in range(1, int(header['TFIELDS']) + 1):
        if f'TSCAL{i}' in header or f'TZERO{i}' in header:
            raise ValueError(f'Scaled FITS columns are not supported: {path}')
        form = str(header[f'TFORM{i}']).strip()
        repeat = int(form[:-1]) if form[:-1] else 1
        code = form[-1]
        if code not in _TYPES:
            raise ValueError(f'Unsupported FITS column format {form!r}: {path}')
        name = str(header.get(f'TTYPE{i}', f'COL{i}')).strip()
        fields.append((name, _TYPES[code], (repeat,)) if repeat != 1 else (name, _TYPES[code]))
    dtype = np.dtype(fields)
    if dtype.itemsize != width:
        raise ValueError(f'Column formats do not add up to NAXIS1 in {path}')
    table = np.frombuffer(data, dtype=dtype, count=rows)
    return {name: table[name].astype(table[name].dtype.newbyteorder('='))
            for name in dtype.names}
