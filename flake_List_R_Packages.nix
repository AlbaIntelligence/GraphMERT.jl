{ pkgs }:
(with pkgs; [
  R
  # To install RAGG
  harfbuzz
  fribidi
  freetype
  libjpeg
  libpng
  libtiff
  libwebp

  # To install shiny.telemwetry
  unixODBC
  unixODBCDrivers.sqlite

]) ++ (with pkgs.rPackages; [
  languageserver
  styler
  # testthat
  # roxygen2

  # Data Processing and Analysis
  # tidyverse
  # gridExtra
  # kableExtra
  # validate

  # Visualization
  # ggplot2
  # plotly
  # viridis
  # corrplot
  # ggcorrplot
  # leaflet
  # heatmaply

  # Machine Learning
  # ROCR
  # ranger
  # VIM
  # caret
  # randomForest
  # nnet

  # Shiny Web Framework
  # shiny
  # shinythemes
  # shinydashboard
  # shinyjs
  # shiny_telemetry
  # shinyWidgets
  # shinyBS

  # Optimization and Constraint Solving
  # GA
  # igraph
  # ompr
  # ROI
  # lpSolve
  # optimx
  # Pareto
  # nsga2R
  # nsga3
  # mco
  # emoa

  # Database Integration
  # RSQLite

  # Testing and Quality Assurance
  # covr
  # lintr

  # Additional Utilities
  # DT
  # R6
  # htmltools
  # jsonlite
  # yaml
])
