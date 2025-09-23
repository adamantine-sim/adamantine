## See NIX.md for help getting started with Nix

{
  description = "Software to simulate heat transfer for additive manufacturing";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    utils.url   = "github:numtide/flake-utils";
  };

  outputs = inputs @ { self, utils, ... }: utils.lib.eachDefaultSystem (system: rec {
    config = rec {
      pkgs = import inputs.nixpkgs {
        inherit system;
        inherit (import ./nix/nixpkgs/config.nix {}) overlays config;
      };
    };

    lib = with config; {
      callPackage = set: pkgs.lib.callPackageWith (pkgs // set);
    };

    derivations = with config; rec {
      callPackage = lib.callPackage libs;

      libs = {
        adiak   = callPackage ./nix/dependencies/adiak   {};
        caliper = callPackage ./nix/dependencies/caliper {};
        arborx  = callPackage ./nix/dependencies/arborx  {};

        dealii = let
          versions = rec {
            latest = v962;
            v962   = callPackage ./nix/dependencies/dealii/v9.6.2.nix { inherit callPackage; };
            v952   = callPackage ./nix/dependencies/dealii/v9.5.2.nix { inherit callPackage; };
          };
        in (versions.latest) // {
          inherit versions;
        };
      };

      adamantine = let
        versions = rec {
          devel = callPackage ./nix/adamantine/common.nix {
            version = self.shortRev or self.dirtyShortRev;
            src     = self;
          };

          stable = v100;

          v100 = callPackage ./nix/adamantine/v1.0.0.nix {
            inherit callPackage;
            dealii = libs.dealii.versions.v952;
          };
        };
      in (versions.devel) // {
        inherit versions;
      };
    };

    packages = rec {
      default = adamantine.versions.devel;

      inherit (derivations) adamantine;
    };

    devShells = with config; rec {
      default = adamantineDev;

      adamantineDev = pkgs.mkShell rec {
        name = "adamantine-dev";

        packages = with pkgs; [
          git
          clang-tools
          ninja
        ] ++ pkgs.lib.optionals (pkgs.stdenv.hostPlatform.isLinux) [
          gdb
          cntr
	];

	inputsFrom = [
	  self.outputs.packages.${system}.default
	];

        # For dev, we want to disable hardening.
        hardeningDisable = [
          "bindnow"
          "format"
          "fortify"
          "fortify3"
          "pic"
          "relro"
          "stackprotector"
          "strictoverflow"
        ];

        # Ensure the locales point at the correct archive location.
        LOCALE_ARCHIVE = pkgs.lib.optional (pkgs.stdenv.hostPlatform.isLinux) (
          "${pkgs.glibcLocales}/lib/locale/locale-archive"
        );
      };
    };
  });

  nixConfig = {
    extra-substituters = [ "https://mdfbaam.cachix.org" ];
    extra-trusted-public-keys = [ "mdfbaam.cachix.org-1:WCQinXaMJP7Ny4sMlKdisNUyhcO2MHnPoobUef5aTmQ=" ];
  };
}
