# Makefile for BTP LaTeX Report

PDFLATEX = pdflatex -interaction=nonstopmode
TARGET = main
OUTPUT = btp_report.pdf

.PHONY: all pdf clean

all: pdf

pdf: $(TARGET).tex
	$(PDFLATEX) $(TARGET).tex
	$(PDFLATEX) $(TARGET).tex
	cp $(TARGET).pdf $(OUTPUT)
	@echo "=== BTP Report PDF Generated Successfully ($(OUTPUT)) ==="
	@pdfinfo $(OUTPUT) | grep Pages

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).toc
