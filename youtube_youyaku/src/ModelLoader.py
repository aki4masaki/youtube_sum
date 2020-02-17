import torch
from torch import nn
from pytorch_pretrained_bert import BertModel
import pdb

from src.models.encoder import TransformerInterEncoder
from src.LangFactory import LangFactory


class Bert(nn.Module):
    def __init__(self, bert_model, temp_dir):
        super(Bert, self).__init__()
        # self.model = BertModel.from_pretrained(bert_model, cache_dir=temp_dir)
        # bert_model = 'bert-base-uncased',  temp_dir = './model/English'
        # ここを model = BertModel.from_pretrained('/content/YouyakuMan/transformers/') にしてみる。
        self.model = BertModel.from_pretrained('/content/YouyakuMan/transformers/')

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask=mask)
        top_vec = encoded_layers[-1]
        return top_vec


class Summarizer(nn.Module):
    def __init__(self, opt, lang):
        super(Summarizer, self).__init__()
        self.langfac = LangFactory(lang)

        self.bert = Bert(self.langfac.toolkit.bert_model, './model/English')

        self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size,
                                               opt['ff_size'],
                                               opt['heads'],
                                               opt['dropout'],
                                               opt['inter_layers'])

    def load_cp(self, pt):
        self.load_state_dict(pt, strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class ModelLoader(Summarizer):
    def __init__(self, cp, opt, lang):
        # cp = 'checkpoint/jp/cp_step_710000.pt'
        # opt = 'checkpoint/jp/opt_step_710000.pt'
        # print(".bin inside eval",self.langfac.toolkit.model.eval())
        # print(".bin inside model.to('cuda')",self.langfac.toolkit.model.to('cuda'))

        cp_statedict = torch.load(cp, map_location=lambda storage, loc: storage)
        # cp_statedict = cp #AttributeError: 'str' object has no attribute 'copy'
        # print("cp_statedict",cp_statedict)
        opt = dict(torch.load(opt))
        super(ModelLoader, self).__init__(opt, lang)
        # print("cp",cp)
        # print("opt",opt)
        # print("cp_statedict",cp_statedict)

        # self.load_cp(cp_statedict) -> error: shape difference
        self.load_cp(cp_statedict)
        self.eval()
