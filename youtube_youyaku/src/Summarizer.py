import torch
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


class Summarizer:
    def __init__(self, test_data, model, n, translator=None):
        self.data = test_data
        self.model = model
        self.translator = translator
        n = self.n_distribution(n)
        start_n = 0

        path_w = 'summarized.txt'
        with open(path_w, mode='w') as f:
            for i, data in enumerate(self.data):
                self._evaluate(data)
                self._extract_n(n[i], start_n)
                start_n += n[i]
                # New txt
                f.write(self.pred[0])
                f.write("。\n")
                # New file

    def n_distribution(self, n):
        # print("len(self.data)",len(self.data))
        if len(self.data) == 1:
            return [n]
        else:
            last_ratio = sum([x > 0 for x in self.data[-1]['src']])/512
            article_len = len(self.data) - 1 + last_ratio
            n_sub = max([n/article_len, 0.51])  # At least 1 summary per data input
            n_extract = [round(n_sub)]*len(self.data)
            n_extract[-1] = round(n_sub*last_ratio)
            return n_extract

    def _extract_n(self, n, start_n):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        _pred = []

        #self.selected_ids[:self.str_len]
        #-> s[:self.str_len * 0.5] で出力長が調整できるか確認
        for j in self.selected_ids[:self.str_len]:
            # print("self.selected_ids",self.selected_ids)
            # print("self.str_len",self.str_len)
            candidate = self.src_str[j].strip()
            # self.src_str = test_data['src_str']
            if not _block_tri(candidate, _pred):
                if len(candidate) > 68:
                    #_block_tri　三文字連続で_pred後に再出現する文字がある時、candidateを棄却
                    # print("candidate",candidate)
                    # print("len(candidate)",len(candidate))
                    _pred.append(candidate)

            if (len(_pred) == n) or (n==0):
                break

        # Translate Summaries to other language
        if self.translator:
            _pred = self.translator.output(_pred)



        # Print result
        preds = []
        for i, pred in enumerate(_pred):
            sys.stdout.write("[要約文%s] %s \n" % (start_n+i+1, pred))
            # print("[Summary %s] %s" % (start_n+i+1, pred))
            preds.append(pred)
        self.pred = preds



    def _evaluate(self, test_data):
        self.model.eval()
        # print("test_data",test_data)
        with torch.no_grad():
            src = torch.tensor([test_data['src']])
            segs = torch.tensor([test_data['segs']])
            clss = torch.tensor([test_data['clss']])
            mask = torch.tensor([test_data['mask']])
            mask_cls = torch.tensor([test_data['mask_cls']])


            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            sent_scores = sent_scores + mask.float()
            selected_ids = torch.argsort(-sent_scores, 1)
            # print("_evaluate.selected_ids",selected_ids)
            selected_ids = selected_ids.cpu().data.numpy()
        self.selected_ids = selected_ids[0]
        # print("test_data",test_data)
        self.src_str = test_data['src_str']
        self.str_len = len(test_data['src_str'])
        self.fname = test_data['fname']

    # Archieve so far
    def diet(self, percent):
        _pred = []
        diet_ids = self.selected_ids[:int(self.str_len * percent)]
        diet_text = [x for i, x in enumerate(self.src_str) if i in diet_ids]
        diet_text = '. \n'.join(diet_text) + '. '
        return diet_text
