#!/bin/bash
# Helper script to start the frontend dev server
# This loads nvm and starts the Vite dev server

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

cd "$(dirname "$0")"
npm run dev

