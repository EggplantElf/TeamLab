import re

def shape(word):
    output = ''
    state = ''
    for c in word:
        if state == '':
            if re.match('[A-Z]', c):
                state = 'A'
            elif re.match('[a-z]', c):
                state = 'a'
        