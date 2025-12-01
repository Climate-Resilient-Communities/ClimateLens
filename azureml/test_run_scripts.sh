#!/bin/bash
set -e

echo "-------------------------------------"
echo " Test Script Running "
echo "-------------------------------------"

echo "Working directory:"
pwd

echo "Listing everything:"
ls -R .

echo "Checking Python:"
python --version

echo "Trying to import modules:"
python - << 'EOF'
print("Python import test:")
import os
print("Files in code/:", os.listdir("code"))
EOF

echo "-------------------------------------"
echo " Test Complete "
echo "-------------------------------------"