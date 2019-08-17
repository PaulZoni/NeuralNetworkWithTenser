import string
import re


def load_doc(file_name):
    file = open(file=file_name, mode='r')
    text = file.read()
    file.close()
    return text


def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = image_desc
    return mapping


def save_doc(descriptions, filename):
    lines = list()
    for key, desc in descriptions.items():
        lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()


def clean_descriptions(descriptions):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    for key, desc in descriptions.items():
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [re_punc.sub('', w) for w in desc]
        desc = [word for word in desc if len(word) > 1]
        descriptions[key] = ' '.join(desc)


filename = '/Users/p.polyakov/Documents/data_set/Flickr8k/Flickr8k_text/Flickr8k.token.txt'
descriptions = load_descriptions(doc=load_doc(file_name=filename))
print('Loaded: %d ' % len(descriptions))
clean_descriptions(descriptions=descriptions)
all_tokens = ' '.join(descriptions.values()).split()
print(len(all_tokens))
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))
print(vocabulary)
#save_doc(descriptions=descriptions, filename='/Users/p.polyakov/Documents/data_set/descriptions.txt')
