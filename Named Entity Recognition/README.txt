The following files have 2 methods for the NER problem, ( notice that as written below both arent useful for state-of-art NER systems, and are here only for learning)
"ner_basic.py" contain a basic solution of getting a "model" of sentences and tags for each word in it, and then when using it for tagging will give the most common tag appeard to each word
	notice that this should not be used in practice as it for only studying purpose as its not a practical method ( thats why i didnt documanted it inline) 
	Example why not to use:  lets say we have the following sentences "i need to book a flight" and " this book is about Gilgamesh", in both cases book will have the same tag even as its easly noticed
		 that in the first sentence book is a verb and in the second its a noun

"ner_bigram_hmm.py" contain a Hidden Markov Model(HMM) which is a  statistical model used to identify and classify named entities and Used the Viterbi algorithm
	In the HMM model, the sequence of words in a sentence is observed, and the sequence of corresponding named entity tags is hidden.
	 HMM uses probabilities for transitions between tags and emissions of words given a tag to determine the most likely sequence of named entity tags for the observed words.
	
	this model is than used to choose a tag via the Viterbi algorithm:
		The Viterbi algorithm finds the most probable sequence of hidden states (tags) given
    		a sequence of observed events (words) and a model of the transition probabilities
    		between states and emission probabilities of observations given states.

the HMM method is quite usefull for it simplicity 
BUT, HMMs have been largely replaced by more advanced models like CRFs and deep learning methods (e.g., LSTM-CRF, BERT) in state-of-the-art NER systems.
 These newer models provide better performance by capturing more complex patterns and dependencies in the data.