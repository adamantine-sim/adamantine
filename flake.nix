## See NIX.md for help getting started with Nix

{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-24.11";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = { nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      adiak = (with pkgs; stdenv.mkDerivation {
        pname = "adiak";
        version = "0.4.0";

        src = fetchgit {
          url = "https://github.com/LLNL/Adiak";
          rev = "v0.4.0";
          sha256 = "sha256-S4ZLU6f/njdZXyoQdCJIDzpQTSmfapZiRe4zIex5f0Q=";
          fetchSubmodules = true;
        };

        buildInputs = [
          cmake
        ];

        cmakeFlags = [
          "-DBUILD_SHARED_LIBS=ON"
          "-DCMAKE_BUILD_TYPE=Release"
        ];
      });

      caliper = (with pkgs; stdenv.mkDerivation {
        pname = "caliper";
        version = "2.10.0";

        src = fetchgit {
          url = "https://github.com/LLNL/caliper";
          rev = "9b5b5efe9096e3f2b306fcca91ae739ae5d00716";
          sha256 = "sha256-4rnPbRYtciohLnThtonTrUBO+d8vyWvJsCgoqbJI5Rg=";
          fetchSubmodules = true;
        };

        buildInputs = [
          cmake
          adiak
          python3
          openmpi
        ];

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DBUILD_SHARED_LIBS=ON"
          "-DWITH_ADIAK=ON"
          "-DWITH_LIBDW=ON"
          "-DWITH_LIBPFM=ON"
          "-DWITH_LIBUNWIND=ON"
          "-DWITH_MPI=ON"
          "-DWITH_SAMPLER=ON"
        ];
      });

      arborx = (with pkgs; stdenv.mkDerivation {
        pname = "arborx";
        version = "1.5";

        src = fetchgit {
          url = "https://github.com/arborx/ArborX";
          rev = "v1.5";
          sha256 = "sha256-qEC4BocPyH9mmU9Ys0nNu8s0l3HQGPHg8B1oNcGwXOQ=";
          fetchSubmodules = true;
        };

        buildInputs = [
          cmake
          openmpi
          trilinos_override
        ];

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DBUILD_SHARED_LIBS=ON"
          "-DCMAKE_CXX_EXTENSIONS=OFF"
          "-DARBORX_ENABLE_MPI=ON"
        ];
      });

      deal_II_952 = (with pkgs; stdenv.mkDerivation {
        pname = "deal_II";
        version = "9.5.2";
        src = fetchgit {
          url = "https://github.com/dealii/dealii";
          rev = "6f07117be556bf929220c50820b4dead54dc31d0";
          sha256 = "sha256-wJIrSuEDU19eZT66MN0DIuSiWQ1/gdu+gHeMYrbQkxk=";
          fetchSubmodules = true;
        };

        buildInputs = [
          cmake
          openmpi
          trilinos_override
          arborx
          p4est
          boost183
        ];

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=DebugRelease"
          "-DCMAKE_CXX_STANDARD=17"
          "-DCMAKE_CXX_EXTENSIONS=OFF"
          "-DDEAL_II_WITH_TBB=OFF"
          "-DDEAL_II_WITH_64BIT_INDICES=ON"
          "-DDEAL_II_WITH_COMPLEX_VALUES=OFF"
          "-DDEAL_II_WITH_MPI=ON"
          "-DDEAL_II_WITH_P4EST=ON"
          "-DP4EST_DIR=${p4est}"
          "-DDEAL_II_WITH_ARBORX=ON"
          "-DARBORX_DIR=${arborx}"
          "-DDEAL_II_WITH_TRILINOS=ON"
          "-DTRILINOS_DIR=${trilinos_override}"
          "-DDEAL_II_TRILINOS_WITH_SEACAS=OFF"
          "-DDEAL_II_COMPONENT_EXAMPLES=OFF"
          "-DDEAL_II_WITH_ADOLC=OFF"
          "-DDEAL_II_ALLOW_BUNDLED=OFF"
         ];
      });

      deal_II_962 = deal_II_952.overrideAttrs ( with pkgs; previousAttrs : rec {
        version = "9.6.2";
        src = previousAttrs.src.override {
          rev = "v9.6.2";
          sha256 = "sha256-YVOQbvzWWSl9rmYd6LBx4w2S8wuxhVF8T2dKdOphta4=";
        };
      });
      
      trilinos_extra_args = ''
        -DTrilinos_ENABLE_ML=ON
        -DBoost_INCLUDE_DIRS=${pkgs.boost183}/include
        -DBoostLib_INCLUDE_DIRS=${pkgs.boost183}/include
        -DBoostLib_LIBRARY_DIRS=${pkgs.boost183}/lib
        -DTPL_ENABLE_BoostLib=ON
      '';
      trilinos_withMPI = pkgs.trilinos.override ( previous: { withMPI = true; boost = pkgs.boost183; });
      trilinos_override = trilinos_withMPI.overrideAttrs ( previousAttrs : rec {
          preConfigure = previousAttrs.preConfigure + ''
                       cmakeFlagsArray+=(${trilinos_extra_args})
                       '';            
          version = "14.4.0";
          src = previousAttrs.src.override {
            rev = "${previousAttrs.pname}-release-${pkgs.lib.replaceStrings [ "." ] [ "-" ] version}";
            sha256 = "sha256-jbXQYEyf/p9F2I/I7jP+0/6OOcH5ArFlUk6LHn453qY=";
          };
        }
      );

      adamantine-base = (with pkgs; stdenv.mkDerivation rec {
        pname = "adamantine";
        version = "1.0";

        src = fetchgit {
          url = "https://github.com/adamantine-sim/adamantine";
          rev = "v1.0";
          sha256 = "sha256-pwwGgk4uIEOkyNLN26nRYvkzQZR53TJW14R9P99E3Ts=";
        };

        buildInputs = [
          arborx
          adiak
          caliper
          cmake
          p4est
          trilinos_override
          boost183
        ] ++ propagatedBuildInputs;

        
        propagatedBuildInputs = [
          openmpi
        ];

        cmakeFlags = [
          "-DADAMANTINE_ENABLE_ADIAK=ON"
          "-DADAMANTINE_ENABLE_CALIPER=ON"
          "-DBOOST_DIR=${boost183}"
        ];

        installPhase = ''
          mkdir -p $out/bin
          cp bin/adamantine $out/bin
        '';

        doCheck = true;
        check = ''
          ctest -R integration_2d
        '';
      });

      adamantine-release = adamantine-base.overrideAttrs ( with pkgs; previousAttrs : rec {
        buildInputs = previousAttrs.buildInputs ++ [ deal_II_952 ];
        cmakeFlags = previousAttrs.cmakeFlags ++ [
          "-DDEAL_II_DIR=${deal_II_952}"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DCMAKE_CXX_FLAGS=-ffast-math"
        ];
      });
      
      adamantine-latest = adamantine-base.overrideAttrs ( with pkgs; previousAttrs : rec {
        version = "latest";
        src = pkgs.lib.cleanSource ./.;
        buildInputs = previousAttrs.buildInputs ++ [ deal_II_962 ];
        cmakeFlags = previousAttrs.cmakeFlags ++ [
          "-DDEAL_II_DIR=${deal_II_962}"
          "-DCMAKE_BUILD_TYPE=Release"
          "-DCMAKE_CXX_FLAGS=-ffast-math"
        ];
      });

      adamantine-debug = adamantine-base.overrideAttrs ( with pkgs; previousAttrs : rec {
        version = "debug";
        separateDebugInfo = true;
        src = pkgs.lib.cleanSource ./.;
        buildInputs = previousAttrs.buildInputs ++ [ deal_II_962 ];
        cmakeFlags = previousAttrs.cmakeFlags ++ [
          "-DDEAL_II_DIR=${deal_II_962}"
          "-DCMAKE_BUILD_TYPE=Debug"
          "-DCMAKE_CXX_FLAGS=-O0"
          "-DCMAKE_CXX_FLAGS_DEBUG=-g3"
        ];
      });
      
      myDebugInfoDirs = pkgs.symlinkJoin {
        name = "myDebugInfoDirs";
        paths = with pkgs; [
          adamantine-debug.debug
        ];
      };
      
    in rec {
      defaultApp = flake-utils.lib.mkApp {
        drv = defaultPackage;
      };
      defaultPackage = adamantine-latest;

      devShells.default = pkgs.mkShell {
        buildInputs = [
          adamantine-latest
        ];
      };

      devShells.release = pkgs.mkShell {
        buildInputs = [
          adamantine-release
        ];
      };

      devShells.debug = pkgs.mkShell {
        NIX_DEBUG_INFO_DIRS = "${pkgs.lib.getLib myDebugInfoDirs}/lib/debug";
        buildInputs = [
          adamantine-debug
          pkgs.gdb
        ];
      };
    }
  );
}
