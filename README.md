# CS224N
These are assignments I did when self-studying CS224N: Natural Language Processing with Deep Learning from Stanford. This course introduces the theoretical aspects of Deep Learning and NLP, from word embeddings, RNNs, to Transformer architectures, pretraining, finetuning, etc. The emphasis on practical implementations is a highlight, where I learned to design and implement Transformer based Language Models, explore architecture designs, and perform finetuning on specific tasks. Python and PyTorch is used throughout the course. 

In each Assignment folder, the code folder includes the code written for the assignment, as well as a "Solution.pdf" file for all written responses to the questions. A summary to all the assignments are listed below

### Assignment 1
- Introduction to word vectors, including prediction-based and count-based word vectors

### Assignment 2 Word2Vec and Dependency Parsing
- Implemented a Neural Transition-Based Dependency Parser using PyTorch. Essentially, it is a neural classifier that aims to iteratively generate the dependency parsing of a sentence.
- Worked through the mathematics behind Word2Vec in depth, gaining both mathematical and intuitive understanding. 
- Learned about training tools and techniques, including ADAM optimizer and dropouts.  
### Assignment 3 Neural Machine Translation with RNNs and NMT Systems
- Implemented a Seq2Seq network with attention, which uses a Bidirectional LSTM Encoder and an Unidirectional LSTM Decoder, for the purpose of Machine Translation. Achieves BLEU score of 19.6. 
- Practiced implementing all key components in PyTorch 
- Learned to use Google Cloud's Virtual Instances of GPU for training
- Analyzed key difficulties and sources of errors behind NMT Systems

### Assignment 4 Self-Attention, Transformers, and Pretraining</b> 
- Performed pretraining and finetuning on the task "accessing the birth place of a notable person". Pretraining+FT shown to outperform significantly FT only model. (The code was originally a fork from Andrej Karpathy's minGPT.)
- Transformer architecture in depth, mathematically exploring the benefits of multi-head attention. We performed
- Learned about Sinusoidal Positional Embedding and Rotary Positional Embedding
    