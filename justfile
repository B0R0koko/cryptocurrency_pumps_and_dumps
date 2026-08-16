set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# The IEEE Access class and Type 1 fonts live in paper/styles, while figures
# live in paper/src/images. The build runs from paper/src.
tex_env := 'TEXINPUTS="../styles//:./images//:" TEXFONTS="../styles//:" T1FONTS="../styles//:"'

format-all:
    poetry run python -m black core features market_data preprocessing backtest test

pylint:
    poetry run python -m pylint core features market_data preprocessing backtest test

mypy:
    poetry run python -m mypy core features market_data preprocessing backtest test

paper:
    cd paper/src && {{tex_env}} pdflatex -interaction=nonstopmode access.tex && {{tex_env}} pdflatex -interaction=nonstopmode access.tex

paper-highlighted:
    cd paper/src && {{tex_env}} pdflatex -interaction=nonstopmode access_highlighted.tex && {{tex_env}} pdflatex -interaction=nonstopmode access_highlighted.tex

presentation:
    cd presentation && lualatex -interaction=nonstopmode main.tex && lualatex -interaction=nonstopmode main.tex
