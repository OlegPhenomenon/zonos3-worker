"""Blank the GPU code of architectures this image never runs on inside CUDA shared libraries.

cuBLAS/cuBLASLt ship cubins for sm_50 … sm_120 (~750 MB compressed in the image) but the workers are Ampere/Ada only
(compute capability 8.6/8.9, the Vast search is compute_cap >= 860 < 900). The CUDA driver picks, per kernel, the best
cubin the device can run (exact arch, else an older minor of the same major), so entries for other majors and for
sm_100/sm_120 are never loaded. We keep sm_80/86/89 (the exact arch plus the fallback chain) and overwrite the payload
of every other entry with zeros: the file keeps its size and layout (no ELF/fatbin rewriting, no relocations), the
entry headers stay valid, and the layer compresses to almost nothing.

Usage: prune_fatbin.py [--keep 80,86,89] [--drop-ptx] LIB...
  --drop-ptx  also blank PTX entries of kept architectures (libggml-cuda carries PTX next to exact cubins)
"""
import argparse, mmap, re, struct, subprocess, sys

MAGIC = bytes.fromhex('50ed55ba')        # 0xBA55ED50, little endian: start of a fat binary container
KIND_PTX, KIND_CUBIN = 1, 2
SECTIONS = ('.nv_fatbin', '.cask_resource')   # cuBLASLt keeps its CASK kernels (sm_100/120 only) in the second


def section_ranges(path):
    out = subprocess.run(['readelf', '-S', '-W', path], capture_output=True, text=True, check=True).stdout
    for name in SECTIONS:
        m = re.search(re.escape(name) + r'\s+PROGBITS\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)', out)
        if m:
            yield name, int(m.group(1), 16), int(m.group(2), 16)


def entries(mm, start, end):
    """(header offset, kind, arch, payload offset, payload size) of every entry of every container in [start, end)."""
    pos = start
    while pos < end:
        i = mm.find(MAGIC, pos, end)
        if i < 0:
            return
        _magic, _version, header_size, fat_size = struct.unpack_from('<IHHQ', mm, i)
        p, q = i + header_size, i + header_size + fat_size
        while p < q:
            kind, _v, entry_header, size, _csz, _u, _minor, _major, arch = struct.unpack_from('<HHIQIIHHI', mm, p)
            yield p, kind, arch, p + entry_header, size
            p += entry_header + size
        pos = q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', default='80,86,89')
    ap.add_argument('--drop-ptx', action='store_true')
    ap.add_argument('libs', nargs='+')
    args = ap.parse_args()
    keep = {int(a) for a in args.keep.split(',')}
    for path in args.libs:
        kept = blanked = 0
        with open(path, 'r+b') as f, mmap.mmap(f.fileno(), 0) as mm:
            for name, off, size in section_ranges(path):
                for _hdr, kind, arch, payload, n in entries(mm, off, off + size):
                    if kind == KIND_CUBIN:
                        drop = arch not in keep
                    elif kind == KIND_PTX:
                        drop = args.drop_ptx or arch > max(keep)
                    else:
                        raise SystemExit(f'{path}: unknown fatbin entry kind {kind}')
                    if drop:
                        mm[payload:payload + n] = bytes(n)
                        blanked += n
                    else:
                        kept += n
            mm.flush()
        print(f'{path}: kept {kept / 1e6:.1f} MB of GPU code, blanked {blanked / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
