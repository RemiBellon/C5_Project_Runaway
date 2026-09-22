#!/bin/bash
# Bloque tout `git push` direct sur main/master.
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [[ "$COMMAND" =~ git[[:space:]]+push ]] && [[ "$COMMAND" =~ (main|master) ]]; then
  echo "Blocked: push direct sur une branche protégée. Passe par une pull request." >&2
  exit 2
fi
exit 0
