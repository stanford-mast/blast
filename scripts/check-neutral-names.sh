#!/usr/bin/env bash
# BLAST's public protocol and source stay vendor-neutral.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pattern='[a]rker'
status=0

if git ls-files | grep -i -e "$pattern"; then
  echo "error: tracked file names above contain a vendor name" >&2
  status=1
fi

if git grep -n -i -a -e "$pattern" -- .; then
  echo "error: tracked files above contain a vendor name" >&2
  status=1
fi

exit "$status"
