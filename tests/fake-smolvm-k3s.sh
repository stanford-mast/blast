#!/bin/bash
set -euo pipefail

machines=${XDG_DATA_HOME:?}/machines
mkdir -p "$machines"
shift
operation=${1:?}
shift

value() {
  local wanted=$1
  shift
  while (($#)); do
    if [[ $1 == "$wanted" ]]; then
      printf '%s\n' "$2"
      return
    fi
    shift
  done
  return 1
}

case "$operation" in
  create)
    name=$(value --name "$@")
    mkdir -p "$machines/$name/fs"
    if source=$(value --from "$@"); then
      tar -xf "$source" -C "$machines/$name"
    else
      printf '%s %s\n' "$(value --cpus "$@")" "$(value --mem "$@")" > "$machines/$name/config"
    fi
    ;;
  start|pause|unpause)
    ;;
  checkpoint)
    name=$(value --name "$@")
    output=$(value --output "$@")
    tar -cf "$output" -C "$machines/$name" .
    ;;
  delete)
    rm -rf "$machines/$(value --name "$@")"
    ;;
  exec)
    name=$(value --name "$@")
    command=${!#}
    guest="$machines/$name/fs/tmp/blast-k3s-state"
    command=${command//\/tmp\/blast-k3s-state/$guest}
    exec sh -c "$command"
    ;;
  ls)
    python3 - "$machines" <<'PY'
import json
import pathlib
import sys

result = []
for machine in pathlib.Path(sys.argv[1]).iterdir():
    cpus, memory = (machine / "config").read_text().split()
    result.append({"name": machine.name, "cpus": int(cpus), "memory_mib": int(memory)})
print(json.dumps(result))
PY
    ;;
esac
