TEX=pdflatex
MAIN=main

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex bibliography.bib figures/*.png
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).nav $(MAIN).out $(MAIN).snm $(MAIN).toc
	rm -f $(MAIN).vrb

distclean: clean
	rm -f $(MAIN).pdf
