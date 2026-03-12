{
  description = "Maiko Emulator Development Environment";

  inputs = {
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    llms.url = "github:numtide/llm-agents.nix";
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs-stable,
      nixpkgs-unstable,
      llms,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs-unstable {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
        pkgs-stable = import nixpkgs-stable {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
        llmsPkgs = llms.packages.${pkgs.stdenv.hostPlatform.system};
      in
      {
        devShells.default = pkgs.mkShell {
          # Use stdenv to get proper C compilation environment
          # This ensures standard C library headers (ctype.h, errno.h, etc.) are available
          stdenv = pkgs.stdenv;

          # Native build inputs (tools needed during build)
          nativeBuildInputs = with pkgs-stable; [
          ];

          # Build inputs (libraries and runtime dependencies)
          buildInputs =
            (with pkgs-stable; [ roswell ])
            ++ (with pkgs; [
              # Basic utilities
              ripgrep # Text search utility
              hexdump # Hexadecimal dump utility
              jq # JSON processor
              xxd # Hexadecimal editor

              # C compiler (both for compatibility)
              # clang # Preferred C compiler
              # gcc # Alternative C compiler (useful for compatibility testing)

              # C standard library headers (needed for ctype.h, errno.h, etc.)
              # stdenv provides CC (C compiler) with proper includes
              # glibc.dev # C standard library development headers

              # Language compilers
              openssl # OpenSSL library (for MCP)
              openspecfun

              # IDEs
              bun # For amp
              code-cursor
              vscode

              rlwrap # To get a history when using sbcl/ecl from the CLI
            ])
            ++ (with llmsPkgs; [
              cursor-agent
              amp
              kilocode-cli
              opencode
              openspec
              spec-kit
            ]);

          # Set up environment variables for pkg-config and C compilation
          shellHook = ''
            # Ensure C standard library headers are available
            export NIX_CFLAGS_COMPILE="-isystem ${pkgs.glibc.dev}/include $NIX_CFLAGS_COMPILE"

            # Set up library paths for runtime linking
            # Nix automatically sets up library paths, but we ensure SDL2/X11 are available
            export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:${
              pkgs.lib.makeLibraryPath (
                with pkgs-stable;
                [
                  openssl
                ]
              )
            }:$LD_LIBRARY_PATH"

            # Set up PKG_CONFIG_PATH
            export PKG_CONFIG_PATH="${
              pkgs.lib.makeSearchPath "lib/pkgconfig" (
                with pkgs;
                [
                ]
              )
            }:$PKG_CONFIG_PATH"
          '';
        };
      }
    );
}
