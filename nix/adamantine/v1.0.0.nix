attrs @ { callPackage, fetchFromGitHub, ... }:

callPackage ./common.nix (
  rec {
    version = "1.0";
    src = fetchFromGitHub {
      owner = "adamantine-sim";
      repo  = "adamantine";
      rev   = "v${version}";
      hash  = "sha256-pwwGgk4uIEOkyNLN26nRYvkzQZR53TJW14R9P99E3Ts=";
    };
  } // attrs
)
