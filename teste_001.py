import nltk

texto = 'Mr.Green killed Colonel Mustard in the study with the candlestick. Mr.Green is not a very nice fellow.'
#print(texto.split('.')) #maneira manual
frases = nltk.tokenize.sent_tokenize(texto) #quebra frases
#print(frases)

tokens = nltk.word_tokenize(texto) #utilizado para a construção de compiladores, usado para identificar o que é cada uma das palvras
#print(tokens)

classes = nltk.pos_tag(tokens) #separa as palavras e mostra a classe gramatical de cada uma link: cs.nyu.edu/grishman/jet/guide/PennPOS.html (Penn Part of Speech Tags)
#print(classes)

entidades = nltk.chunk.ne_chunk(classes) #identiifica entidades
print(entidades)

#Notas:
    #https://www.youtube.com/watch?v=siVUal-TeMc -> aula base
    #nltk.org
    #poemas com IA
    #Google Brain