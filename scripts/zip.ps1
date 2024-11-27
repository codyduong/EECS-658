$toZip = $args[0].Trim('.\').Trim('\')
$path = "CodyDuong_$toZip"

# Cleanup old zips
Remove-Item -Recurse -Force $path -ErrorAction SilentlyContinue
Remove-Item -Force "$path.zip" -ErrorAction SilentlyContinue
Get-ChildItem -Path $toZip -Exclude ".venv" | ForEach-Object {
  $destination = Join-Path -Path $path -ChildPath $_.Name
  Copy-Item -Recurse -Path $_.FullName -Destination $destination
}
Compress-Archive -Path $path -DestinationPath "$path.zip"

# Cleanup temp
Remove-Item -Recurse -Force $path -ErrorAction SilentlyContinue