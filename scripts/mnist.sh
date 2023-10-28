#!/bin/bash

# Get the image directory
current_directory="/home/jovyan/data/mnist"

# Iterate through all .png files in the current directory and its subdirectories
find "$current_directory" -type f -name "*.png" | while read -r filepath; do
    # Extract the file name without extension
    file_name=$(basename "$filepath" .png)
    
    # Extract the parent folder name
    folder_name=$(basename "$(dirname "$filepath")")

    # Construct the new filename
    new_filename="${file_name}_${folder_name}.png"
    
    # Create the full path to the new file in the current directory
    new_filepath="$current_directory/$new_filename"
    
    # Rename the file
    mv "$filepath" "$new_filepath"
    
    echo "Renamed: $filepath to $new_filepath"
done