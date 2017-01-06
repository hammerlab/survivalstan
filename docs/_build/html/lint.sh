#!/bin/bash
set -o errexit

find stanmodels test -name '*.py' \
    | xargs pylint \
            --errors-only \
            --disable=print-statement

echo 'Passes pylint check'
