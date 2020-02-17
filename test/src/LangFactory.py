import re
from pyknp import Juman
from configparser import ConfigParser
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertConfig, BertModel

import pdb

config = ConfigParser()
config.read('./config.ini')


class JumanTokenizer:
    def __init__(self):
        print("welcome to JumanTokenizer")
        # print("remove from Juman(): command=jumanpp,option= --model= jumanpp/model/jumandic.jppmdl --config = jumanpp/model/jumandic.conf.in")
        self.juman = Juman()
        # print("self.juman",self.juman)

    def __call__(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class LangFactory:
    def __init__(self, lang):
        self.support_lang = ['en', 'jp']
        self.lang = lang
        self.stat = 'valid'
        if self.lang not in self.support_lang:
            print('Language not supported, will activate Translation.')
            self.stat = 'Invalid'
        self._toolchooser()
        print("self.toolkit",self.toolkit)

    def _toolchooser(self):
        if self.lang == 'jp':
            self.toolkit = JapaneseWorker()
        elif self.lang == 'en':
            self.toolkit = EnglishWorker()
        else:
            self.toolkit = EnglishWorker()


class JapaneseWorker:
    def __init__(self):
        self.juman_tokenizer = JumanTokenizer()
        print("JapaneseWorker started at Langfac")


        # config = BertConfig.from_json_file('checkpoint/jp/bert_config.json')
        # print("config",config)




        # "config['DEFAULT']['vocab_path'] -> vocab.text paste"
        self.bert_tokenizer = BertTokenizer("checkpoint/jp/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/vocab.txt",
                                            do_lower_case=False, do_basic_tokenize=False)
        self.cls_id = self.bert_tokenizer.vocab['[CLS]']
        self.mask_id = self.bert_tokenizer.vocab['[MASK]']
        print("self.cls_id",self.cls_id)

        self.bert_model = 'bert-base-uncased' #'PATH_TO_BERTJPN'

        self.cp = 'checkpoint/jp/cp_step_710000.pt'
        self.opt = 'checkpoint/jp/opt_step_710000.pt'

    @staticmethod
    def linesplit(src):
        """
        :param src: type str, String type article
        :return: type list, punctuation seperated sentences
        """
        def remove_newline(x):
            x = x.replace('\n', '')
            return x

        def remove_blank(x):
            x = x.replace(' ', '')
            return x

        def remove_unknown(x):
            unknown = ['\u3000']
            for h in unknown:
                x = x.replace(h, '')
            return x
        src = remove_blank(src)
        src = remove_newline(src)
        src = remove_unknown(src)
        src_line = re.split('。(?<!」)|！(?<!」)|？(?!」)', src)
        src_line = [x for x in src_line if x is not '']
        return src_line

    def tokenizer(self, src):
        """
        :param src: type list, punctuation seperated sentences
        :return: token: type list, numberized tokens
                 token_id: type list, tokens
        """
        token = []
        token_id = []

        def _preprocess_text(text):
            return text.replace(" ", "")  # for Juman

        print("src",src)
        print(len(src))


        for sentence in src:

            preprocessed_text = _preprocess_text(sentence)
            print("_preprocess_text",preprocessed_text)

            juman_tokens = self.juman_tokenizer(preprocessed_text)
            print("juman_tokens",juman_tokens)
            tokens = self.bert_tokenizer.tokenize(" ".join(juman_tokens))
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            print("tokens",tokens)
            ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            token += tokens
            token_id += ids
            print("token",token)
            print("token_id",token_id)
        print("token",token)
        print("token_id",token_id)
        return token, token_id


class EnglishWorker:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cls_id = self.bert_tokenizer.vocab['[CLS]']
        self.mask_id = self.bert_tokenizer.vocab['[MASK]']
        self.bert_model = 'bert-base-uncased'

        self.cp = 'checkpoint/en/stdict_step_300000.pt'
        self.opt = 'checkpoint/en/opt_step_300000.pt'

    @staticmethod
    def linesplit(src):
        def remove_newline(x):
            x = x.replace('\n', ' ')
            return x

        def replace_honorifics(x):
            honors = ['Mr', 'Mrs']
            for h in honors:
                x = x.replace(h + '. ', h + ' ')
            return x

        src = remove_newline(src)
        src = replace_honorifics(src)
        src_line = re.split('\.', src)
        src_line = [x for x in src_line if x is not '']
        return src_line

    def tokenizer(self, src):
        token = []
        token_id = []

        for sentence in src:
            tokens = self.bert_tokenizer.tokenize(sentence)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            token += tokens
            token_id += ids
        return token, token_id
