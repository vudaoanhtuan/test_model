import visen
import re
import numpy as np

def to_teencode_1(sent):
    sent = visen.remove_tone(sent)
    sent = sent.lower()
    tokens = sent.split()
    for i,word in enumerate(tokens):
        if word.find('y') >= 0:
            if word.find('ay') >= 0:
                pass
            elif word.find('uy') >= 0:
                word = word.replace('uy', 'y')
            else:
                word = word.replace('y', 'i')
        word = re.sub("ch$", "x", word)
        word = re.sub("^ph", "f", word)
        # word = re.sub("^k","c", word)
        word = re.sub("^kh", "k", word)
        word = re.sub("^gi", "j", word)
        word = re.sub("^gh", "g", word)
        word = re.sub("^ngh", "ng", word)
        word = re.sub("^qu", "q", word)
        word = re.sub("ng$", "g", word)
        word = re.sub("nh$", "h", word)
        #word = word.replace('h', 'k')
        word = word.replace('i', 'j')
        tokens[i] = word
    return ' '.join(tokens)


def to_teencode_2(sent):
    sent = visen.remove_tone(sent)
    sent = sent.lower()
    tokens = sent.split()
    for i,word in enumerate(tokens):
        word = word.replace("a", "4")
        word = word.replace("e", "3")
        word = word.replace("g", "q")
        if word[0] == "":
            pass
        else:
            word = word.replace("kh", "kh")
            word = word.replace("h", "k")
        word = word.replace("i", "j")
        word = word.replace("o", "0")
        tokens[i] = word
    return ' '.join(tokens)

def to_teencode_3(sent):
    sent = sent.lower()
    tokens = sent.split()
    for i,word in enumerate(tokens):
        word = re.sub("b", '|3', word)
        word = re.sub("đ", '+)', word)
        word = re.sub("d", '])', word)
        word = re.sub("l", '|_', word)
        word = re.sub("á|ắ|ấ", '4\'', word)
        word = re.sub("à|ằ|ầ", '4`', word)
        word = re.sub("a|ă|â", '4', word)
        word = re.sub("ả|ẳ|ẩ", '4', word)
        word = re.sub("ạ|ặ|ậ", '4', word)
        word = re.sub("ã|ẵ|ẫ", '4', word)
        word = re.sub("é|ế", '3\'', word)
        word = re.sub("ẻ|ể", '3?', word)
        word = re.sub("è|ề", '3`', word)
        word = re.sub("e|ê", '3', word)
        word = re.sub("ẽ|ễ", '3', word)
        word = re.sub("i", 'j', word)
        word = re.sub("ì", 'j`', word)
        word = re.sub("í", 'j\'', word)
        word = re.sub("ỉ", 'j', word)
        word = re.sub("ĩ", 'j', word)
        word = re.sub("ỏ|ổ|ở", '0', word)
        word = re.sub("o|ô|ơ", '0', word)
        word = re.sub("ò|ồ|ờ", '0`', word)
        word = re.sub("ó|ố|ớ", '0\'', word)
        word = re.sub("õ|ỗ|ỡ", '0', word)
        word = re.sub("ọ|ộ|ợ", '0.', word)
        word = re.sub("ú|ứ", 'u\'', word)
        word = re.sub("ù|ừ", 'u`', word)
        word = re.sub("ủ|ử", 'u', word)
        word = re.sub("ũ|ữ", 'u', word)
        word = re.sub("ụ|ự", 'u.', word)
        word = re.sub("u|ư", 'u', word)
        word = re.sub("y", 'ij', word)
        tokens[i] = word
    return ' '.join(tokens)


def transform_teencode(sent):
    words = sent.split()
    for i,w in enumerate(words):
        func = np.random.choice([to_teencode_1, to_teencode_2, to_teencode_3])
        w = func(w)
        words[i] = w
    return ' '.join(words)

if __name__ == "__main__":
    sent = 'xin chào thế giới trong một vũ trụ này'
    print(to_teencode_1(sent))
    print(to_teencode_2(sent))
    print(to_teencode_3(sent))
    print(transform_teencode(sent))