#!/usr/bin/env bash
# deny-scan.sh <file>...: fail when any file matches $DENY, the PUBLISH_DENY_REGEX secret (a
# case-insensitive grep -E regex of names that must never be published). Binary files are
# scanned through `strings`. It prints only WHERE (file:line, or the file of a binary), never
# the matched text or the regex: secret masking hides the whole secret, not a line that
# contains part of it. Used by ci.yml, release.yml and text-gate.yml.
set -euo pipefail
[ -n "${DENY:-}" ] || { echo "::error::DENY is empty: set the PUBLISH_DENY_REGEX repository secret"; exit 1; }
rc=0; printf '\n' | grep -qiE -e "$DENY" 2>/dev/null || rc=$?
[ "$rc" = 1 ] || { echo "::error::PUBLISH_DENY_REGEX is not a usable grep -E regex (it errors, or matches empty text)"; exit 1; }
[ $# -gt 0 ] || { echo "::error::deny-scan: nothing to scan"; exit 1; }

hits=0
for f in "$@"; do
    [ -f "$f" ] || { echo "::error::deny-scan: $f is not a file"; exit 1; }
    if [ -s "$f" ] && ! grep -qI '' "$f"; then   # what grep calls binary (images, wasm, a NUL)
        # Not grep -q: an early exit SIGPIPEs strings, and pipefail would read that as no match.
        if strings -n 4 "$f" | grep -iE -e "$DENY" >/dev/null; then
            echo "::error::deny-list match in $f (binary)"; hits=$((hits + 1))
        fi
    else
        for n in $(grep -niE -e "$DENY" -- "$f" | cut -d: -f1); do
            echo "::error::deny-list match at $f:$n"; hits=$((hits + 1))
        done
    fi
done
if [ "$hits" -gt 0 ]; then
    echo "$hits deny-list match(es) in $# scanned file(s). The text is not shown. Remove it from the file, commit message, tag message or PR/issue text."
    exit 1
fi
echo "deny-list scan clean: $# file(s)"
