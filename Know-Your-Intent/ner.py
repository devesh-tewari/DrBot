import spacy
nlp = spacy.load('en')
parsed_text = nlp(u"After brushing, I had blood and swollen gums.")

#get token dependencies
for text in parsed_text:
    #subject would be
    if text.dep_ == "nsubj":
        subject = text.orth_
    #iobj for indirect object
    indirect_object=None
    if text.dep_ == "iobj":
        indirect_object = text.orth_
    #dobj for direct object
    if text.dep_ == "dobj":
        direct_object = text.orth_

print(subject)
print(direct_object)
print(indirect_object)
