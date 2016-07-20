#!/bin/bash
set -o errexit

find survivalstan test -name '*.py' \
    | xargs pylint \
            --errors-only \
            --disable=print-statement \
            --extension-pkg-whitelist=numpy \
            --extension-pkg-whitelist=patsy

echo 'Passes pylint check'
