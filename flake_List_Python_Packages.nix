{ pkgs, pythonVersion }:
let
  python =
    if pythonVersion == "3.11" then
      pkgs.python311
    else if pythonVersion == "3.12" then
      pkgs.python312
    else if pythonVersion == "3.13" then
      pkgs.python313
    else
      pkgs.python3;
in
(with pkgs; [
  uv
  poetry
])
++ [
  (python.withPackages (
    ps: with ps; [
      # marker-pdf # See https://github.com/datalab-to/marker
      streamlit
      # streamlit-ace
      # jupyter
      # jupyterlab
      # numpy
      # scipy
      # pandas
      # matplotlib
      # scikit-learn
      # tox
      # pygments
    ]
  ))
]
