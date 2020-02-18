# youtube_sum

 [![License](https://poser.pugx.org/ali-irawan/xtra/license.svg)](*https://poser.pugx.org/ali-irawan/xtra/license.svg*)

### Introduction

This is youtube summarization (speech to text) tweeting machine.

これは youtube の内容を要約した動画紹介テキストを Twitter で Tweet するコードです。

using [youtube-dl](https://github.com/ytdl-org/youtube-dl/tree/067aa17edf5a46a8cbc4d6b90864eddf051fa2bc) as Download model.
using IBM Watson as speech to text model.
using [YouyakuMan](https://github.com/neilctwu/YouyakuMan) as summatization model, extract top N important sentences.

---
### Prerequisites (from YouyakuMan)

#### General requirement(originaly from YouyakuMan, changed by aki4masaki)

```
pip install torch
pip install pytorch_pretrained_bert
pip install googletrans
pip install pyknp
pip install watson-developer-cloud>=1.4.0
```

#### Japanese specific requirement (from YouyakuMan)

- [BERT日本語Pretrainedモデル — KUROHASHI-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル)
- [Juman++ V2の開発版](https://github.com/ku-nlp/jumanpp)[ — KUROHASHI-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル)


---

### Pretrained Model (from YouyakuMan)

English: [Here](https://drive.google.com/open?id=1wxf6zTTrhYGmUTVHVMxGpl_GLaZAC1ye)

Japanese: [Here](https://drive.google.com/open?id=10hJX1QBAHfJpErG2I8yhcAl2QB_q28Fi)

Download and put under directory `checkpoint/en` or `checkpoint/jp`

---

### How to use Example

please check [youtube_youyaku.ipynb](https://github.com/aki4masaki/youtube_sum/blob/master/youtube_youyaku.ipynb)

#### Note


---
### Version Log:

2020-02-17  First version
