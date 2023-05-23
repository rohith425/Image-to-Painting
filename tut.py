# run this from a normal command line

import spacy
# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_md')
# process a sentence using the model
doc = nlp("This is some text that I am processing with Spacy")
# It's that simple - all of the vectors and words are assigned after this point
# Get the vector for 'text':
# Get the mean vector for the entire sentence (useful for sentence classification etc.)
print(len(doc.vector))

print('end')