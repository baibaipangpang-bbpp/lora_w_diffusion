#!/bin/bash

# prepend 书法 to all txt files, this allows to mix different styles of caligraphy
# run this command when curating data set to diffusion models. 


# Detect OS type
os_type=$(uname)

echo "Detected OS: $os_type"
echo "Prepending '书法 ' to all .txt files in current directory..."

# iterate through all .txt files
for file in *.txt; do
  [ -f "$file" ] || continue  # Skip if no txt files

  if [[ "$os_type" == "Darwin" ]]; then
    # macOS (BSD sed)
    sed -i '' '1s/^/书法 /' "$file"
  elif [[ "$os_type" == "Linux" ]]; then
    # Linux (GNU sed)
    sed -i '1s/^/书法 /' "$file"
  else
    echo "Unsupported OS: $os_type"
    exit 1
  fi
done

echo "Done."