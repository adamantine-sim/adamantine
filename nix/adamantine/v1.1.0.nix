attrs @ { callPackage, fetchFromGitHub, ... }:

callPackage ./common.nix (
  rec {
    version = "1.1";
    src = fetchFromGitHub {
      owner = "adamantine-sim";
      repo  = "adamantine";
      rev   = "v${version}";
      hash  = "sha256-m7mUdd3wLhWlFtPdgx9JePCU/wBt78v/a7jhSYTfA10=";
    };
  } // attrs
)
