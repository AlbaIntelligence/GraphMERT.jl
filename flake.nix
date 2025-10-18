{
  description = "GraphMERT";

  # inputs = {
  #   # NixOS official package source, here using the nixos-unstable branch
  #   nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # };

  outputs = { nixpkgs, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = { allowUnfree = true; };
      };

      commandName = "envFHS";

      enableJulia = true;
      juliaVersion = "1.12.0";

      enablePython = false;
      pythonVersion = "3.12";

      enableNodeJS = false;

      # Doc libraries
      enableQuarto = false;

      # Graphical libraries
      enableGraphical = true;
      enableNVIDIA = false;

      #
      # Basic packages used in all environments
      standardPackages = ps:
        (with ps; [
          # Utilities
          curl
          ncurses
          openspecfun
          openssl
          unzip
          utillinux
          which

          # Building
          patch
          autoconf
          binutils
          clang
          cmake
          expat
          gcc
          gfortran
          gmp
          gnumake
          gperf
          libxml2
          m4
          nss
          stdenv.cc

          # DB
          # sqlite

          # IDE
          code-cursor
          cursor-cli
          vscode
        ]);

      # Graphical packages used in all environments
      graphicalPackages = ps:
        (with ps; [
          alsa-lib
          at-spi2-atk
          at-spi2-core
          atk
          cairo
          cups
          dbus
          expat
          ffmpeg
          fontconfig
          freetype
          gettext
          glfw
          glib
          glib.out
          gtk3
          jupyter-all
          libGL
          libcap
          libdrm
          libgpg-error
          libnotify
          libpng
          libsecret
          libuuid
          libxkbcommon
          ncurses
          nspr
          nss
          pango
          pango.out
          pdf2svg
          systemd
          vulkan-loader
          vulkan-headers
          vulkan-validation-layers
          wayland # for Julia
          zlib
        ]) ++ (with ps.xorg; [
          libICE
          libSM
          libX11
          libXScrnSaver
          libXcomposite
          libXcursor
          libXcursor
          libXdamage
          libXext
          libXfixes
          libXi
          libXinerama
          libXrandr
          libXrender
          libXt
          libXtst
          libXxf86vm
          libxcb
          libxkbfile
          xorgproto
        ]);

      nvidiaPackages = ps:
        (with ps; [
          cudatoolkit_11
          cudnn_cudatoolkit_11
          linuxPackages.nvidia_x11
        ]);

      # quartoPackages = ps: let
      #   quarto = ps.callPackage ./scientific_nix/quarto.nix {rWrapper = null;};
      # in [quarto];

      pythonPackages = ps:
        (ps.callPackage ./flake_List_Python_Packages.nix {
          pkgs = ps;
          pythonVersion = pythonVersion;
        });

      targetPkgs = ps:
        (standardPackages ps)
        ++ ps.lib.optionals enableGraphical (graphicalPackages ps)
        ++ ps.lib.optionals enableJulia [
          (ps.callPackage ./scientific_nix/julia.nix {
            juliaVersion = juliaVersion;
          })
          # ps.julia
          ps.openspecfun
        ] ++ ps.lib.optionals enableQuarto [ ps.quarto ]
        ++ ps.lib.optionals enableNVIDIA (nvidiaPackages ps)
        ++ ps.lib.optionals enablePython (pythonPackages ps)
        ++ ps.lib.optionals enableNodeJS
        (with ps; [ nodejs nodePackages.npm nodePackages.yarn ]);

      std_envvars = ''
        export EXTRA_CCFLAGS="-I/usr/include"
        export FONTCONFIG_FILE=/etc/fonts/fonts.conf
        export LIBARCHIVE=${pkgs.libarchive.lib}/lib/libarchive.so
      '';

      juliaEnvvars = ''
        export LD_LIBRARY_PATH=${pkgs.openspecfun}/lib:${pkgs.zlib}/lib::${pkgs.curl}/lib:$LD_LIBRARY_PATH
      '';

      graphical_envvars = ''
        export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale
      '';

      nvidia_envvars = ''
        export CUDA_PATH=${pkgs.cudatoolkit_11}
        export LD_LIBRARY_PATH=${pkgs.cudatoolkit_11}/lib:${pkgs.cudnn_cudatoolkit_11}/lib:${pkgs.cudatoolkit_11.lib}/lib:${pkgs.zlib}/lib::${pkgs.curl}/lib:$LD_LIBRARY_PATH
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
      '';

      envvars = std_envvars
        + pkgs.lib.optionalString enableGraphical graphical_envvars
        + pkgs.lib.optionalString enableNVIDIA nvidia_envvars
        + pkgs.lib.optionalString enableJulia juliaEnvvars;

      # FHS environment package
      envFHS = pkgs.buildFHSEnv {
        name = commandName;

        targetPkgs = targetPkgs;
        # multiPkgs = pkgs: (with pkgs; [ zlib ]);

        runScript = "zsh"; # default is bash
        profile = envvars;

        # Misc extras
        extraOutputsToInstall = [ "man" "dev" ];
      };
    in {
      defaultPackage.x86_64-linux = envFHS;
      packages.x86_64-linux.envFHS = envFHS;
      devShells.x86_64-linux.default = envFHS;

      nixpkgs.config.allowUnfree = true;

      # Development shell for nix develop and direnv
      # devShells.x86_64-linux.default = pkgs.mkShell pythonVersion{
      #   buildInputs = targetPkgs pkgs;
      #   shellHook = envvars;
      # };
      #
      nixosModules.default = import ./scientific_nix/module.nix;
    };
}
