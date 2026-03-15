{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  llmsPkgs = inputs.llms.packages.${pkgs.stdenv.hostPlatform.system};
  # CPU-only llama.cpp (no CUDA) for dev environment; reuse via Nix store on subsequent entry
  # nixpkgs llama-cpp: only cudaSupport is overridable; CPU build is default when cudaSupport = false
  llamaCppCpu = pkgs.llama-cpp.override { cudaSupport = false; };
in
{
  # https://devenv.sh/basics/
  # env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages =
    (with pkgs; [
      # utils
      git

      # IDE
      bash # Required by Speckit scripts
      vscode
      copilot-cli
      copilot-language-server

      # LLM - CPU-only llama.cpp (CLI + server); workflow uses llama-cpp instead of ollama
      llamaCppCpu

      # Python - Hugging Face Hub CLI (e.g. huggingface-cli download for GGUF / encoder weights)
      python313Packages.huggingface-hub
    ])
    ++ (with llmsPkgs; [
      cursor-agent
      gemini-cli # Broken build
      kilocode-cli
      opencode
      openskills
      openspec
      spec-kit
    ]);

  # https://devenv.sh/languages/
  languages.julia.enable = true;

  # https://devenv.sh/processes/
  # Run llama-cpp server (llama-server) for local inference; workflow uses llama-cpp instead of ollama.
  processes.llama-server = {
    exec = ''
      ${llamaCppCpu}/bin/llama-server --host 127.0.0.1 --port 8080
    '';
    ready = {
      http.get = {
        port = 8080;
        path = "/";
      };
      initial_delay = 2;
    };
  };

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo Welcome to your DEVENV development environment.
    echo
  '';

  # https://devenv.sh/basics/
  enterShell = ''
    hello         # Run scripts directly
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
    # CPU-only llama.cpp: assert binary on PATH when present (nixpkgs may expose as llama or llama-cli)
    command -v llama >/dev/null 2>&1 || command -v llama-cli >/dev/null 2>&1 || echo "Note: llama.cpp not on PATH yet (run devenv up or devenv shell to build)"
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
