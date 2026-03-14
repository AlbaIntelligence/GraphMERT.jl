{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  llmsPkgs = inputs.llms.packages.${pkgs.stdenv.hostPlatform.system};
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
      code-cursor
      amp-cli
      bash # Required by Speckit scripts
      vscode

      # LLM - Ollama for local inference (CPU only; cudaSupport disabled in devenv.yaml)
      ollama-cpu
      # vllm
      # llama-cpp  # Disabled: pulls in CUDA by default; enable only if you set up CUDA or use override
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
  # Run ollama as a process. (services.ollama is a NixOS-only option; devenv has no built-in ollama service.)
  processes.ollama = {
    exec = ''
      export OLLAMA_MODELS="''${HOME:-/tmp}/.ollama/models"
      export OLLAMA_KEEP_ALIVE="1h"
      export OLLAMA_LLM_LIBRARY="cpu"
      export OLLAMA_HOST="127.0.0.1:11434"
      ${lib.getExe pkgs.ollama-cpu} serve
    '';
    ready = {
      http.get = { port = 11434; path = "/"; };
      initial_delay = 2;
    };
  };
  # open-webui is not a built-in devenv service; run it manually or add a process if needed.

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
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
