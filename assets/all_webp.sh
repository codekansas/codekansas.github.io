#!/bin/bash
# Converts any PNGs or JPEGs in the post-images directory to WEBPs.
# Requires imagemagick.

set -e

# root_dir=$(pwd)/post-images/
root_dir=$(dirname "$0")/post-images/

for file in $(find $root_dir -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg"); do
    echo "Converting $file to WEBP..."
    cwebp -q 80 $file -o "${file%.*}.webp"
    rm $file
done
