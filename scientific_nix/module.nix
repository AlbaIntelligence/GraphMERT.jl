{ config, lib, pkgs, ... }:
with lib;
let cfg = config.programs.classroom-fhs;
in {
  options.programs.classroom-fhs = {
    enable = mkEnableOption "an FHS for science!";

    juliaVersions = mkOption {
      type = types.listOf (types.submodule {
        options = {
          version = mkOption { type = types.str; };
          default = mkOption {
            type = types.bool;
            default = false;
          };
        };
      });
      default = [{
        version = "1.11.6";
        default = true;
      }];
    };

    enableNVIDIA = mkOption {
      type = types.bool;
      default = false;
    };

    enableGraphical = mkOption {
      type = types.bool;
      default = false;
    };
  };

  config = {
    home.packages = builtins.concatMap (version-spec:
      let
        classroom-fhs = pkgs.callPackage scientific-nix/fhs.nix {
          enableNVIDIA = cfg.enableNVIDIA;
          enableGraphical = cfg.enableGraphical;
          juliaVersion = version-spec.version;
        };

        fhsCommand = commandName: commandScript:
          classroom-fhs.override {
            inherit commandName;
            inherit commandScript;
          };

        name = "julia"
          + (if version-spec.default then "" else "-" + version-spec.version);

        python = if version-spec.default then [
          (fhsCommand "python3" "python3")
          (fhsCommand "python" "python3")
          (fhsCommand "poetry" "poetry")
        ] else
          [ ];

        quarto = if version-spec.default then
          [ (fhsCommand "quarto" "quarto") ]
        else
          [ ];
      in [ (fhsCommand name "julia") (fhsCommand "${name}-bash" "bash") ]
      ++ python ++ quarto) cfg.juliaVersions;
  };
}
