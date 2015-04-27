from sentence import *
from model import *
import sys


if __name__ == '__main__':
    for sent in read_sentence(sys.argv[1]):
        print sent