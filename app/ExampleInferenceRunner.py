import SentimentAnalysis


print('Starting textproc')
textProc = SentimentAnalysis.TextProcessor()


textSample1 = "The unfortunate details of the hate crime have been released today, Wednesday 3rd November"
sentimentResult1 = textProc.evaluate_text(textSample1)

textSample2 = "The winners of the tournament will recieve a selection of prizes"
sentimentResult2 = textProc.evaluate_text(textSample2)

print ("Sentiment processing - input phrase 1:")
print(textSample1)
print (sentimentResult1)


print ("Sentiment processing - input phrase 2:")
print(textSample2)
print (sentimentResult2)