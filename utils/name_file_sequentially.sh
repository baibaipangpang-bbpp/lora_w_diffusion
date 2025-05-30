i=1

# Loop over image files (common extensions)
for file in *.jpg *.jpeg *.png *.bmp *.gif; do
    # Skip if no matching files (e.g., literal *.jpg)
    [ -e "$file" ] || continue

    # Get the extension
    ext="${file##*.}"

    # Rename to sequential number
    mv -- "$file" "$i.$ext"
    i=$((i + 1))
done
