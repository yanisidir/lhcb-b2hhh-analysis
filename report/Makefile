TEX=pdflatex
BIB=bibtex
MAIN=main

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex bibliography.bib figures/*.png
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf
