"""Fetch a checksum-verified portable Julia runtime for validation only."""

import hashlib
from pathlib import Path
import urllib.request
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]/".validation"
    root.mkdir(exist_ok=True)
    archive = root/"julia.zip"
    url = "https://julialang-s3.julialang.org/bin/winnt/x64/1.13/julia-1.13.1-win64.zip"
    expected = "648e39c58158c5c8dde08bb0df0b5bce51d517be1396a37a539bb61bc241c8b4"
    urllib.request.urlretrieve(url, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Julia archive checksum mismatch")
    with zipfile.ZipFile(archive) as package:
        package.extractall(root)
    print(root/"julia-1.13.1"/"bin"/"julia.exe")


if __name__ == "__main__":
    main()
