#!/bin/bash
# Start the persistent audio worker service

cd "$(dirname "$0")"

echo "Starting Audio Worker Service..."
echo "Press Ctrl+C to stop"
echo ""

# Use --gpu flag if you have CUDA GPU available
# python audio_worker_service.py --gpu

python audio_worker_service.py
