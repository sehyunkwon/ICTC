#!/bin/bash

# Set the directory path
directory="/home/jovyan/data/stanford-40-actions/JPEGImages"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Directory not found: $directory"
  exit 1
fi

# Iterate through the files in the directory
for file in "$directory"/*; do
  if [ -f "$file" ]; then
    # Extract the label and number from the original filename
    original_filename=$(basename "$file")
    filename_no_extension="${original_filename%.*}"
    
    # Split the filename by underscores and get the last part as the number
    IFS='_' read -ra parts <<< "$filename_no_extension"
    num_parts=${#parts[@]}
    
    if (( num_parts < 2 )); then
      echo "Skipping $original_filename - Invalid format"
      continue
    fi
    
    number="${parts[$num_parts - 1]}"
    label="${parts[@]:0:$num_parts-1}"
    label=$(IFS='_'; echo "${label[*]}")  # Join label parts with underscores
    
    # Create the new filename
    new_filename="${number}_${label}.jpg"

    # Rename the file
    mv "$file" "$directory/$new_filename"
    echo "Renamed $original_filename to $new_filename"
  fi
done