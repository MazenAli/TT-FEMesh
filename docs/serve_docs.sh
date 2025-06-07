#!/bin/bash

# Build the documentation
make html

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "Starting local server..."
    echo "Open http://localhost:8000 in your web browser"
    echo "Press Ctrl+C to stop the server"
    
    # Start the server
    cd _build/html
    python -m http.server 8000 --bind 0.0.0.0
else
    echo "Error: Documentation build failed"
    exit 1
fi 