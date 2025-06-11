# 	Fantastic Labels and Where to Find Them: Attention-Based Label Selection for Text-to-Text Classification

Generative language models, particularly adopting text-to-text frameworks, have shown significant success in NLP tasks. While much research has focused on input representations via prompting techniques, less attention has been given to optimizing output representations. Previous studies found inconsistent effects of label representations on model performance in classification tasks using these models. In this work, we introduce a novel method for selecting well-performing label representations by leveraging the attention mechanisms of Transformer models. We used an Italian T5 model fine-tuned on a topic classification task, trained on posts extracted from online forums and categorized into 11 classes, to evaluate different label representation selection strategies. We’ve employed a context-mixing score called Value Zeroing to assess each token’s impact to select possible representations from the training set. Our results include a detailed qualitative analysis to identify which label choices most significantly affect classification outcomes, suggesting that using our approach to select label representations can enhance performance.

## Cite
```
 @inproceedings{
            papucci2024fantastic,
            title={Fantastic Labels and Where to Find Them: Attention-Based Label Selection for Text-to-Text Classification.},
            author={Papucci, Michele and Miaschi, Alessio and Dell'Orletta, Felice},
            booktitle={NL4AI@ AI* IA},
            year={2024}
          } 
```
