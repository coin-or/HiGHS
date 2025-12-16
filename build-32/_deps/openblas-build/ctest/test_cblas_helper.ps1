$ErrorActionPreference = "Stop"
Get-Content $args[1] | & $args[0]
