# Lrage-Language-Models-Course-Projects
Large language models (LLMs) course projects at University of Tehran (Spring 2024)

## CA1

### Part 1 - Word Embeddings and Masked Language Models

Working with Glove of [Gensim](https://pypi.org/project/gensim/) and examining mask language modeling (particularly BERT)

### Part 2 - Transfer Learning with BERT

Fine-tuning [BERT](https://arxiv.org/abs/1810.04805) for text classification and question answering tasks

## CA2

### Part 1 - GPT

Working with [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model and examining effect of batch normalization

### Part 2 - Soft Prompt Tuning

Soft prompt tuning BERT manually and using [OpenDelta](https://arxiv.org/abs/2307.03084) library for polarity classification of Persian texts

## CA3

### Part 1 - Chain-of-Thought (CoT)

Examining zero-shot [CoT](https://arxiv.org/abs/2201.11903) and implemention of [Self-Consistency](https://arxiv.org/abs/2203.11171) for [Phi-2](https://huggingface.co/microsoft/phi-2) model

### Part 2 - Parameter-Efficient Fine-Tuning (PEFT)

Fine-tuning Phi-2 model with [LORA](https://arxiv.org/abs/2106.09685) for question generation task

### Part 3 - Retrieval-Augmented Generation (RAG)

Examining TF-IDF and semantic retrievers ([FIASS](https://ai.meta.com/tools/faiss/) vector DB) of [LangChain](https://www.langchain.com/) for [Llama 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model and creating a complete RAG chain

## CA4

### Part 1 - Reinforcement Learning from Human Feedback (RLHF)

Supervised fine-tuning GPT-2 model for text summarization, training a reward model and examining PPO fine-tuning

### Part 2 - Quantization and Instruction Tuning

Fine-tunining [Misteral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) model with [QLora](https://arxiv.org/abs/2305.14314) to follow the instructions and examining [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model

### Part 3 - Evaluation Text Using a Language Model

Evaluating text using manually implemented and official [BERTScore](https://arxiv.org/abs/1904.09675)