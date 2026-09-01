#!/usr/bin/env bash
# Recompute the published simulation results in every language here.
#
# Every number in this repository comes out of one Python simulation. The tests
# check that the simulation runs, not that it is right, and a simulation is
# exactly the kind of code where a wrong answer still looks plausible. So each
# implementation below re-derives something published, most of them by drawing
# their own random numbers rather than reading the Python's.
#
# Each is skipped with a message if its toolchain is missing, so a partial
# install still runs the rest. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else printf 'FAILED: %s\n' "$name"; fail=$((fail + 1)); fi
}

# SQL asserts inside the query and prints a verdict column, so a non-ok verdict
# anywhere is the failure signal.
check_sql () {
    local out
    out=$(sqlite3 -init verify/lalonde.sql :memory: "" 2>&1) || return 1
    printf '%s\n' "$out"
    ! printf '%s' "$out" | grep -qiE '\|(not ok|fail)'
}

check_c ()    { cc -std=c99 -O2 -Wall -Wextra -Wpedantic -o "$tmp/cuped" verify/cuped_kernel.c -lm && "$tmp/cuped" "$root"; }
check_go ()   { ( cd verify/gocheck && go run . -root "$root" ); }
check_java () { javac -d "$tmp/j" verify/Pocock.java && java -cp "$tmp/j" Pocock "$root"; }
check_rust () { ( cd verify/peekmc && cargo run --release --quiet -- "$root" ); }

run "SQL, the LaLonde estimate table"        sqlite3 check_sql
run "C, the CUPED variance kernel"           cc      check_c
run "Go, results files and derived columns"  go      check_go
run "R, the peeking and mSPRT rates"         Rscript Rscript verify/verify.R "$root"
run "Java, the Pocock boundary"              javac   check_java
run "Ruby, the prose against the files"      ruby    ruby verify/readme_claims.rb "$root"
run "Node, post-treatment mediation"         node    node verify/post_treatment.mjs "$root"
run "Rust, Monte Carlo error of the rates"   cargo   check_rust

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
