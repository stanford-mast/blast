#!/bin/bash
set -euo pipefail

root=${XDG_DATA_HOME:?}/machines
mkdir -p "$root"
shift
command=${1:?}
shift

value() {
  local wanted=$1
  shift
  while (($#)); do
    if [[ $1 == "$wanted" ]]; then
      echo "$2"
      return
    fi
    shift
  done
  return 1
}

case "$command" in
  create)
    name=$(value --name "$@")
    if from=$(value --from "$@"); then
      cp "$from" "$root/$name"
    else
      cpus=$(value --cpus "$@")
      memory=$(value --mem "$@")
      printf '%s %s\n' "$cpus" "$memory" > "$root/$name"
    fi
    ;;
  start|pause|unpause)
    ;;
  checkpoint)
    name=$(value --name "$@")
    output=$(value --output "$@")
    cp "$root/$name" "$output"
    ;;
  delete)
    name=$(value --name "$@")
    rm -f "$root/$name"
    ;;
  exec)
    sleep 3
    ;;
  ls)
    python3 - "$root" <<'PY'
import json
import pathlib
import sys

machines = []
for path in pathlib.Path(sys.argv[1]).iterdir():
    cpus, memory = path.read_text().split()
    machines.append({"name": path.name, "cpus": int(cpus), "memory_mib": int(memory)})
print(json.dumps(machines))
PY
    ;;
esac
