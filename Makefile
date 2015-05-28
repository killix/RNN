all : 

training.txt : training.txt.zip
	md5sum --check training.txt.md5 || unzip $<

trunk/word2vec : 
	svn checkout http://word2vec.googlecode.com/svn/trunk/
	$(MAKE) -C trunk

vectors.bin : trunk/word2vec training.txt
	trunk/word2vec -train training.txt -output $@ -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15


