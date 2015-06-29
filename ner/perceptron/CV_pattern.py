import re
def ConsVowelPattern(word):
        CVP_NC = word.lower()
        paterns = [['[bcdfghjklmnpqrstwxyz]', 'C'], ['[aeiou]', 'V']]

        for i in paterns:
            CVP_NC = re.sub(i[0],i[1],CVP_NC)
        
        symbols = ['C', 'V']
        CVP_C = CVP_NC
        for i in symbols:
            CVP_C = re.sub(2*i+"+",2*i,CVP_C)
        return CVP_NC, CVP_C


print ConsVowelPattern("Blabalabbabababababa")
