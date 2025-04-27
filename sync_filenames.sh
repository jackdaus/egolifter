#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# sync_filenames.sh – Copy any missing files from A → B and delete extras in B
# Usage:   ./sync_filenames.sh /path/to/dirA /path/to/dirB
# Notes:
#   • Works recursively (sub‑directories, hidden “dot” files, etc.).
#   • Only filenames are guaranteed to match; existing files in B are left
#     untouched even if their contents differ from A (see comment below).
###############################################################################

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <directory‑A> <directory‑B>" >&2
    exit 1
fi

src=$(realpath "$1")
dst=$(realpath "$2")

# 1) Copy anything that exists in A but not in B
# 2) Delete anything that exists in B but not in A
#    --delete       : remove extraneous files/dirs from B
#    --ignore-existing : don’t overwrite files that are already present in B
#    -a             : archive mode (recursive + keeps perms, timestamps, etc.)
#    -v             : verbose so you see what’s happening
rsync -av --ignore-existing --delete "${src}/" "${dst}/"

echo "✓  '${dst}' now has the same set of filenames as '${src}'."

###############################################################################
# Want a dry run first?  Add --dry-run right after -av:
#   rsync -av --dry-run --ignore-existing --delete "${src}/" "${dst}/"
# That prints the actions without making changes.
#
# If you DO want contents updated as well (not just filenames), simply remove
# the --ignore-existing flag so rsync overwrites any differing files in B.
###############################################################################
