param(
    [string]$IpAddress = "172.25.144.4",
    [string]$OutputDir = "certs",
    [int]$ValidDays = 3650,
    [int]$KeySize = 2048
)

$ErrorActionPreference = "Stop"

$targetDir = Resolve-Path -LiteralPath "." | ForEach-Object { Join-Path $_.Path $OutputDir }
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$keyPath = Join-Path $targetDir "server.key"
$certPath = Join-Path $targetDir "server.crt"

$rsa = [System.Security.Cryptography.RSA]::Create($KeySize)
$subject = [System.Security.Cryptography.X509Certificates.X500DistinguishedName]::new("CN=$IpAddress")
$request = [System.Security.Cryptography.X509Certificates.CertificateRequest]::new(
    $subject,
    $rsa,
    [System.Security.Cryptography.HashAlgorithmName]::SHA256,
    [System.Security.Cryptography.RSASignaturePadding]::Pkcs1
)

$san = [System.Security.Cryptography.X509Certificates.SubjectAlternativeNameBuilder]::new()
$san.AddIpAddress([System.Net.IPAddress]::Parse($IpAddress))
$san.AddIpAddress([System.Net.IPAddress]::Parse("127.0.0.1"))
$san.AddDnsName("localhost")
$request.CertificateExtensions.Add($san.Build())
$request.CertificateExtensions.Add(
    [System.Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($false, $false, 0, $false)
)
$request.CertificateExtensions.Add(
    [System.Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new(
        [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature -bor
        [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyEncipherment,
        $false
    )
)

$notBefore = [System.DateTimeOffset]::Now.AddMinutes(-5)
$notAfter = $notBefore.AddDays($ValidDays)
$cert = $request.CreateSelfSigned($notBefore, $notAfter)

function ConvertTo-Pem {
    param(
        [string]$Label,
        [byte[]]$DerBytes
    )
    $base64 = [System.Convert]::ToBase64String($DerBytes)
    $lines = for ($i = 0; $i -lt $base64.Length; $i += 64) {
        $length = [System.Math]::Min(64, $base64.Length - $i)
        $base64.Substring($i, $length)
    }
    return "-----BEGIN $Label-----`n$($lines -join "`n")`n-----END $Label-----`n"
}

function Join-Bytes {
    param([byte[][]]$Parts)
    $length = 0
    foreach ($part in $Parts) {
        $length += $part.Length
    }
    $buffer = New-Object byte[] $length
    $offset = 0
    foreach ($part in $Parts) {
        [System.Buffer]::BlockCopy($part, 0, $buffer, $offset, $part.Length)
        $offset += $part.Length
    }
    return $buffer
}

function New-DerLength {
    param([int]$Length)
    if ($Length -lt 128) {
        return [byte[]]@([byte]$Length)
    }
    $bytes = New-Object System.Collections.Generic.List[byte]
    $value = $Length
    while ($value -gt 0) {
        $bytes.Insert(0, [byte]($value -band 0xff))
        $value = $value -shr 8
    }
    return [byte[]](@([byte](0x80 -bor $bytes.Count)) + $bytes.ToArray())
}

function New-DerInteger {
    param([byte[]]$Value)
    $start = 0
    while ($start -lt ($Value.Length - 1) -and $Value[$start] -eq 0) {
        $start++
    }
    $content = $Value[$start..($Value.Length - 1)]
    if (($content[0] -band 0x80) -ne 0) {
        $content = [byte[]](@([byte]0) + $content)
    }
    return Join-Bytes @([byte[]]@([byte]0x02), (New-DerLength $content.Length), [byte[]]$content)
}

function New-DerSequence {
    param([byte[]]$Content)
    return Join-Bytes @([byte[]]@([byte]0x30), (New-DerLength $Content.Length), $Content)
}

function New-RsaPrivateKeyDer {
    param([System.Security.Cryptography.RSAParameters]$Parameters)
    $content = Join-Bytes @(
        (New-DerInteger ([byte[]]@([byte]0))),
        (New-DerInteger $Parameters.Modulus),
        (New-DerInteger $Parameters.Exponent),
        (New-DerInteger $Parameters.D),
        (New-DerInteger $Parameters.P),
        (New-DerInteger $Parameters.Q),
        (New-DerInteger $Parameters.DP),
        (New-DerInteger $Parameters.DQ),
        (New-DerInteger $Parameters.InverseQ)
    )
    return New-DerSequence $content
}

$certPem = ConvertTo-Pem "CERTIFICATE" $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
$keyPem = ConvertTo-Pem "RSA PRIVATE KEY" (New-RsaPrivateKeyDer $rsa.ExportParameters($true))

[System.IO.File]::WriteAllText($certPath, $certPem, [System.Text.Encoding]::ASCII)
[System.IO.File]::WriteAllText($keyPath, $keyPem, [System.Text.Encoding]::ASCII)

Write-Host "Created HTTPS certificate:"
Write-Host "  cert: $certPath"
Write-Host "  key:  $keyPath"
