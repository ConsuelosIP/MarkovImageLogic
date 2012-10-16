JPEGS=$(wildcard Stanford/*.jpg)
HOGS=$(patsubst Stanford/%.jpg, %-hog.npy, $(JPEGS))
PROC=. venv/bin/activate && ./hog.py

all : $(HOGS)

%-hog.npy : Stanford/%.jpg
	$(PROC) $< $@

clean :
	rm -f *.png *.pgm *.hog *.npy
