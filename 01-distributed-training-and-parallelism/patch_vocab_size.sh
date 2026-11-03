#!/bin/sh

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
patch -p1 -d "${SITE_PACKAGES}" < torchtitan_llama3_debug_vocab.patch

