#!/bin/bash
set -e
cd "$(dirname "$0")"
cd results
/usr/bin/env python3 _create-all-plots.py
