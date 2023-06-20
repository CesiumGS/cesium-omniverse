#! /usr/bin/python3

"""
Intended to be called in the
This script updates the certificates used for any requests that use the core
Context class. While some certs are available on the system, they may not be
consistent or updated. This ensures all certs are uniform and up to date
see: https://github.com/CesiumGS/cesium-omniverse/issues/306
"""

import requests
import sys
import os


def main():
    # --- establish source/destination for certs ---
    if len(sys.argv) < 2:
        print("must provide a filepath for the updated certs")
        return -1

    CERT_URL = "https://curl.se/ca/cacert.pem"
    CERT_FILE_PATH = sys.argv[1]

    # --- ensure directory structure exists ----
    os.makedirs(os.path.dirname(CERT_FILE_PATH), exist_ok=True)

    # --- fetch and write the cert file ----

    req = requests.get(CERT_URL)

    if req.status_code != 200:
        print(f"failed to fetch certificates from {CERT_URL}")
        return -1

    # explicit encoding is required for windows
    with open(CERT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(req.text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
