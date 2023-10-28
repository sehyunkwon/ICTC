#!/bin/bash
# Function to rename image files in a directory
rename_images() {
    local folder="$1"
    local label="$2"
    
    # Initialize a universal counter (outside of this function)
    counter="$3"

    # Iterate through the image files in the directory
    for file in "$folder"/*.png; do
        if [ -f "$file" ]; then
            label="$(basename "$label")"
            # Rename the file with the specified format
            new_name="${directory_path}/${counter}_${label}.png"
            cp "$file" "$new_name"
            
            # Increment the universal counter
            counter=$((counter + 1))
        fi
    done
}

# Initialize a universal counter

directory_path="/home/jovyan/data/cifar100"
counter=1

# Main loop to traverse nested directories
for dir1 in "$directory_path"/*; do
    if [ -d "$dir1" ]; then
        label="$dir1"  # Use the first folder name as the label
        for dir2 in "$dir1"/*; do
            if [ -d "$dir2" ]; then
                echo "$dir2"
                # Rename image files in the current directory
                rename_images "$dir2" "$label" "$counter"
            fi
        done
    fi
done