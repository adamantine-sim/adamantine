# Nix Install

First install the [Nix package manager][NIX] and then enable
[Flakes]. See [nix.dev][[nix.dev] for more help with Nix. To install
without cloning from this repository use,

    $ nix develop github:adamantine-sim/adamantine
	
to install the latest version on main. To install the lastest release use,

    $ nix develop github:adamantine-sim/adamantine#release
    
To install from a working copy use,

    $ nix develop
   
[NIX]: https://nixos.org/download.html
[Flakes]: https://nixos.wiki/wiki/Flakes
[nix.dev]: https://nix.dev
