from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs

def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem

def preprocess_train(domain):
    print( '\t'+domain+' set ...')
    f_label = codecs.open('../../data/raw_txts/'+domain+'_train_label.txt', 'r', 'utf-8')
    f_raw = codecs.open('../../data/raw_txts/'+domain+'_train_raw.txt', 'r', 'utf-8')
    out_label = codecs.open('../../data/processed_txts/'+domain+'_train_label_processed.txt', 'w', 'utf-8')
    out_raw = codecs.open('../../data/processed_txts/'+domain+'_train_raw_processed.txt', 'w', 'utf-8')
    # out = codecs.open('../preprocessed_data/'+domain+'/train.txt', 'w', 'utf-8')

    for line in f_label:
        token_list = line.split(",")
        for tokens in token_list:
            tokens = parseSentence(line)
            if len(tokens) > 0:
                out_label.write(' '.join(tokens)+'\n')

    for line in f_raw:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out_raw.write(' '.join(tokens)+'\n')

def preprocess_test(domain):
    print( '\t'+domain+' set ...')
    f_label = codecs.open('../../data/raw_txts/'+domain+'_test_label.txt', 'r', 'utf-8')
    f_raw = codecs.open('../../data/raw_txts/'+domain+'_test_raw.txt', 'r', 'utf-8')
    out_label = codecs.open('../../data/processed_txts/'+domain+'_test_label_processed.txt', 'w', 'utf-8')
    out_raw = codecs.open('../../data/processed_txts/'+domain+'_test_raw_processed.txt', 'w', 'utf-8')
    # out = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')

    for line in f_label:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out_label.write(' '.join(tokens)+'\n')

    for line in f_raw:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out_raw.write(' '.join(tokens)+'\n')
# def preprocess_test(domain):
#     # For restaurant domain, only keep sentences with single 
#     # aspect label that in {Food, Staff, Ambience}

#     f1 = codecs.open('../datasets/'+domain+'/test.txt', 'r', 'utf-8')
#     f2 = codecs.open('../datasets/'+domain+'/test_label.txt', 'r', 'utf-8')
#     out1 = codecs.open('../preprocessed_data/'+domain+'/test.txt', 'w', 'utf-8')
#     out2 = codecs.open('../preprocessed_data/'+domain+'/test_label.txt', 'w', 'utf-8')

#     for text, label in zip(f1, f2):
#         label = label.strip()
#         if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
#             continue
#         tokens = parseSentence(text)
#         if len(tokens) > 0:
#             out1.write(' '.join(tokens) + '\n')
#             out2.write(label+'\n')

# def preprocess(domain):
#     preprocess_train(domain)
#     # print( '\t'+domain+' test set ...')
#     # preprocess_test(domain)

print( 'Preprocessing raw review sentences ...')
preprocess_train('cures')
preprocess_test('cures')
preprocess_train('causes')
preprocess_test('causes')
preprocess_train('prevents')
preprocess_test('prevents')