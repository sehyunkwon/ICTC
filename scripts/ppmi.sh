#!/bin/bash

directory_path="/home/jovyan/data/ppmi"
# Initialize a running counter
counter=1

# Main loop to traverse files
for file in "$directory_path"/Norm_Play_*_*_*.jpg; do
    if [ -f "$file" ]; then
        # Extract the LABEL from the file name
        label=$(echo "$file" | awk -F'_' '{print $3}')
        # Rename the file with the specified format
        new_name="${counter}_${label}.jpg"
        mv "$file" "$directory_path/$new_name"
        
        # Increment the counter
        counter=$((counter + 1))
    fi
done