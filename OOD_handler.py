import pandas as pd
import classify_intent

replies = pd.read_csv('Datasets/OOD Dataset.tsv', sep='\t')

def get_reply(user_text):
    OOD_intent = classify_intent.get_intent_OOD('what do you look like')
    k = random.randint(0,4)
    possible_replies = list(replies[OOD_intent])
    reply = possible_replies[k]
    return reply
