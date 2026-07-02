MAIN = main

.PHONY: all clean

all:
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	biber $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	$(RM) $(MAIN).aux $(MAIN).bbl $(MAIN).bcf $(MAIN).blg $(MAIN).log
	$(RM) $(MAIN).out $(MAIN).run.xml $(MAIN).toc
