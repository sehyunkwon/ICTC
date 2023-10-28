custom_path="/home/jovyan/data/ppmi/2_categories/"

# Check if the specified directory exists
if [ ! -d "$custom_path" ]; then
    echo "Directory not found: $custom_path"
    exit 1
fi

# Change to the specified directory
cd "$custom_path"

# Loop through all files in the current directory with the format "{number}_{instrument}.jpg"
for file in [0-9]*_*.jpg; do
    # Extract the instrument name from the filename
    instrument=$(echo "$file" | cut -d'_' -f2 | cut -d'.' -f1)

    # Check if the instrument is in the wind instrument list
    if [[ "$instrument" == "Saxophone" || "$instrument" == "Trumpet" || "$instrument" == "Flute" ]]; then
        category="wind"
    # Check if the instrument is in the string instrument list
    elif [[ "$instrument" == "Guitar" || "$instrument" == "Cello" || "$instrument" == "Violin" || "$instrument" == "Harp" ]]; then
        category="string"
    else
        # If the instrument is not in either list, skip this file
        continue
    fi

    # Rename the file using the new category
    new_name=$(echo "$file" | sed "s/$instrument/$category/")
    mv "$file" "$new_name"
    echo "Renamed $file to $new_name"
done