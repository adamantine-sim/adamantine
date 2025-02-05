## Nix

First install the [Nix package manager][NIX] and then enable [Flakes][Flakes].
Alternatively, check out the [Determinate Systems Installer][Determinate] for
an out of the box experience. See [nix.dev][nix.dev] for more help with Nix.

To get a shell with adamantine temporarily installed, run:

    $ nix shell github:adamantine-sim/adamantine
    # Adamantine now available
    $ adamantine --help

To install this permanently, run:

    $ nix profile install github:adamantine-sim/adamantine

To get the latest stable release, use:

    $ nix shell github:adamantine-sim/adamantine#adamantine.stable

To build from a working copy use `nix develop` and run CMake manually:

    $ nix develop
    $ cmake -B build -GNinja
    $ cmake --build build

## direnv

This repository also supports `direnv` for integration with both your shell and
tools like VSCode.

First install direnv from either your distro or via Nix:

    # Via apt...
    $ sudo apt install direnv
    # ... or nix.
    $ nix profile install direnv

Setup direnv for your shell. Tutorials for various shells can be found
[here][DirenvHook]. For bash:

    $ echo "eval \"\$(direnv hook bash)\"" >> ~/.bashrc

Restart your shell and then allow direnv:

    $ cd path/to/my/adamantine
    $ direnv allow

This will automatically enter the nix development shell whenever you enter the
adamantine directory.

If you use VSCode, a great extension that adds direnv support can be found
[here][DirenvVSCode].


[NIX]: https://nixos.org/download.html
[Flakes]: https://nixos.wiki/wiki/Flakes
[nix.dev]: https://nix.dev
[Determinate]: https://github.com/DeterminateSystems/nix-installer
[DirenvHook]: https://direnv.net/docs/hook.html
[DirenvVSCode]: https://marketplace.visualstudio.com/items?itemName=mkhl.direnv
