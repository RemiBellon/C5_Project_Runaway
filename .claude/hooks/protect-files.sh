#!/bin/bash
# Bloque l'édition de fichiers sensibles (secrets, verrous de dépendances, .git/).
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
FILE_PATH="${FILE_PATH//\\//}"

PROTECTED_PATTERNS=(".env" ".git/" "package-lock.json" "uv.lock")

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "Blocked: $FILE_PATH correspond au motif protégé '$pattern'" >&2
    exit 2
  fi
done
exit 0
