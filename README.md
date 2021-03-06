# French NLP Toolkit

State of the art toolkit for Natural Language Processing in French based on CamemBERT/FlauBERT.

Citation:

```
@misc{hadoop,
  author={Wang Xiaoou},
  title={frenchnlp: state of the art toolkit for Natural Language Processing in French based on CamemBERT/FlauBERT},
  year={2021},
  howpublished={\url{https://github.com/xiaoouwang/frenchnlp}},
}
```

```
Wang Xiaoou. (2021). frenchnlp: state of the art toolkit for Natural Language Processing in French based on CamemBERT/FlauBERT. https://github.com/xiaoouwang/frenchnlp.
```

- sentence similarity measure

    * For why average pooling/[cls] shouldn't be used to represent sentence, see

    Reimers, Nils, and Iryna Gurevych. “Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks.” ArXiv:1908.10084 [Cs], August 27, 2019. http://arxiv.org/abs/1908.10084.

    * For use of sentence similarity in real life, see

    Xiaoou Wang, Xingyu Liu, Yimei Yue. “Mesure de similarité textuelle pour l’évaluation automatique de copies d’étudiants.” TALN-RECITAL 2021. [Download](https://xiaoouwang.github.io/xowang/TALN-RECITAL_2021_paper_74.pdf)

- to do, text classification pipelines

## How to use the package

```python
from frenchnlp import *
from transformers import AutoTokenizer, AutoModel
import torch
```
## Transformer-based sentence similarity measure (using CamemBERT as example)

### Using the [cls] token

`compare_compare_cls(model,tokenizer,sentence1,sentence2)`

```py
fr_tokenizer = AutoTokenizer.from_pretrained('camembert-base')
fr_model = AutoModel.from_pretrained('camembert-base')

sentences = [
    "J'aime les chats.",
    "Je déteste les chats.",
    "J'adore les chats."
]

for i in range(1,3):
    print(f"similarité sémantique entre\n{sentences[0]}\n{sentences[i]}")
    print(bert_compare_cls(fr_model,fr_tokenizer,sentences[0],sentences[i]))
```

Output:

```
similarité sémantique entre
J'aime les chats.
Je déteste les chats.
0.9145417
similarité sémantique entre
J'aime les chats.
J'adore les chats.
0.9809468
```

### Average pooling

`compare_bert_average(model,tokenizer,sent1,sent2)`

```python
fr_tokenizer = AutoTokenizer.from_pretrained('camembert-base')
fr_model = AutoModel.from_pretrained('camembert-base')

for i in range(1,3):
    print(f"similarité sémantique entre\n{sentences[0]}\n{sentences[i]}")
    print(compare_bert_average(fr_model,fr_tokenizer,sentences[0],sentences[i])
```

Output:

```
similarité sémantique entre
J'aime les chats.
Je déteste les chats.
0.9145417
similarité sémantique entre
J'aime les chats.
J'adore les chats.
0.9809468
```

### Using multilingual sentence embeddings

See above for the reference on multilingual sentence embeddings.

`compare_sent_transformer(model,sent1,sent2)`

```
from sentence_transformers import SentenceTransformer

sent_model = SentenceTransformer('stsb-xlm-r-multilingual')

for i in range(1,3):
    print(f"similarité sémantique entre\n{sentences[0]}\n{sentences[i]}")
    print(compare_sent_transformer(sent_model,sentences[0],sentences[i])
```

Output:

```
similarité sémantique entre
J'aime les chats.
Je déteste les chats.
0.46124768
similarité sémantique entre
J'aime les chats.
J'adore les chats.
0.9557947
```

## License

### Codes

`frenchnlp` is licensed under Apache License 2.0. You can use `frenchnlp` in your commercial products for free. We would appreciate it if you add a link to `frenchnlp` on your website.

### Models

Unless otherwise specified, all models in `frenchnlp` are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).