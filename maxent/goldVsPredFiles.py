


dev = open("../data/ner/dev.iob", 'r')
predicted = open("predict.iob", 'r')
comparison = open("comparison.iob",'w')
error = open("error.iob",'w')
i = 0
for line in dev:
    if line != '\n':
        #print predicted.readline().strip().split("\t")
        #print line.strip().split("\t")
        line = line.strip() + "\t" + predicted.readline().strip().split("\t")[1] + "\n" 
        comparison.write(line)
    else:
	predicted.readline()
        comparison.write("\n")
        '''
    else:
        '''
dev.close()
predicted.close()
comparison.close()
