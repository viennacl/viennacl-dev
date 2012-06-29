
FILE=matlab-viennacl

# ----------------------------------------------------------------------------

build: build-latex build-dvi build-pdf build-clean
	mv $(FILE).pdf ../

clean: build-clean

# ----------------------------------------------------------------------------

build-latex:
	latex ${FILE}.tex
	latex ${FILE}.tex
	bibtex ${FILE}
	latex ${FILE}.tex	
	bibtex ${FILE}
	latex ${FILE}.tex		
	
build-dvi:
	dvips -Ppdf -G0 -ta4 $(FILE).dvi 
	
build-pdf:
	ps2pdf  -dPDFSETTINGS=/prepress -dCompatibilityLevel=1.3 \
           -dMAxSubsetPct=100 -dSubsetFonts=true \
           -dEmbedAllFonts=true -sPAPERSIZE=a4 \
           -dAutoRotatePages=/None -dOptimize=true $(FILE).ps

build-clean:
	rm -f *.dvi *.log *.aux *.idx *.ps *.out *.toc *~ *.backup
	
