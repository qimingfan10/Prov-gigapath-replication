while IFS=$'\t' read -r uuid filename md5 size state; do
  if [[ "$uuid" != "id" ]]; then
    echo "Downloading $filename ($uuid)"
    curl -O -J "https://api.gdc.cancer.gov/data/$uuid" -o "$filename"
    echo "Finished downloading $filename"
  fi
done < gdc_manifest.txt