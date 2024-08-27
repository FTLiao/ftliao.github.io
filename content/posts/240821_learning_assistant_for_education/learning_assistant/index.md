---
title: 'Learing Assistant for Education'
date: 2024-08-21T13:05:44+02:00
draft: false
params:
    math: true
---

<!-- # Learing Assistant for Education -->

There has been much development of ML in EduTech. One notable effort as an example is Andrej Karpathy's effort in building a startup aiming to apply an AI learning assistant to education ([link to news](https://techcrunch.com/2024/07/16/after-tesla-and-openai-andrej-karpathys-startup-aims-to-apply-ai-assistants-to-education/)). 

In this post, I will share my notes on such a learning assistant.

I will begin by defining my understanding of learning, followed by outline the properties of a learning assistant that effectively supports a learner's educational journey. Subsequently, I will propose an implementation approach for an ML-based learning assistant and conclude by highlighting future research areas in machine learning that are essential for developing such an assistant in a scalable and cost-effective manner.

## Towards Utopia
**The necessity of an automated Learning Assistant**\
I believe that learning is a process of continuous practice and reflection where one becomes aware of their limitations, pushes beyond them, and remains openess to unknowns. Ultimately, through this process, one finds fulfillment.

To equate learning and education for the society as a whole, there is a growing need for an automated learning assistant. For students, it offers personalized support that caters to their unique learning styles, enabling them to master subjects more effectively and confidently. Educators benefit by gaining insights into students' progress, allowing them to focus on more nuanced and human aspects of teaching. On a broader scale, society gains by nurturing a more knowledgeable, adaptable, and skilled population, capable of innovating and solving complex problems in an increasingly complex world.

## Properties of a Learning Assistant
I surmise that such an learning assistant capable of helping an inquisitive mind in mastering subjects and learning should bear the following properties:

1. **Facilitating Learning-to-learn**\
    The assistant encourages learners to engage in active learning, helping them develop transferable strategies that can be applied across various learning contexts. This approach enables learners to use acquired knowledge in structured pathways to solve real-world problems.

2. **Accurate and Reliable in Response**\
    The assistant should provide accurate, reliable, diverse learning materials suitable to the student's needs and learning style, recommending resources based on their progress and adjusting as they advance. 

3. **Personalized Learning**\
    As learners' needs evolve, the assistant, equipped with memory of their learning history, provides tailored feedback and answers to student queries. This feedback is adjusted to their current level of knowledge and aspirations.

Machine learning offers transformative potential in revolutionizing education by creating adaptive, personalized learning assistants that foster growth and deep understanding of knowledge. A promising realization of such a learning assistant could be built upon Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG). This approach would leverage the vast knowledge and conversational capabilities of LLMs while using RAG to provide up-to-date, factual information and personalized content retrieval. 

## Possible Research Directions in ML

To develop a learning assistant that embodies the aforementioned properties, several critical areas in machine learning must be addressed, including, but not limited to, evaluation methods, training data quality, model architecture, reinforcement learning with human feedback (RLHF), and retrieval mechanisms as a form of slow memory. In the following section, I outline promising research directions in these areas.

**Facilitating Learning-to-Learn**\
Learning how to learn empowers individuals to continually adapt to new information and skills throughout their lives. However, the role of machine learning in facilitating the acquisition of such meta-learning skills remains underexplored. Prior work by [4] incorporates high level pedagogical principles into the evaluation of LLMs across various pedagogical dimensions. One such dimension, Analogies, offers a partial assessment of an LLM's ability to instruct learners on how to learn. The work, however, is insufficient as dimension such as abstract thinking and self-discovery could also be critical attributes for evaluating a learning assistant's effectiveness.

To bridge this gap, future research could focus on developing a benchmark, created in collaboration with educators and students, that evaluates learning assistants on their ability to guide learners in discovering transferable principles across diverse tasks without explicit instruction. These tasks could be synthetic and might include challenges such as:
- Learning a systematic method for remembering long numbers that’s transferable to remembering long strings
- Learning to learn in breadth within a constrained timeframe when presented with abstrusive books

**Accurate and Reliable in Response**\
A robust learning assistant should provide accurate and informative responses to learners. However, LLMs, when used as the engine of a learning assistant, are known to hallucination and fail to extract knowledge despite having seen it in pre-training. LLMs often answer questions directly, offering little to no indication of their confidence in the accuracy of their responses. This poses a risk of misleading learners, particularly those unfamiliar with the subject matter.

Addressing these issues in future research is crucial for developing a trustworthy learning agent. Improving the accuracy of knowledge extraction in LLM responses could be achieved by incorporating a diverse set of synthetic question-answer pairs related to key knowledge points into the pre-training data, as demonstrated by [2]. Enhancing the provision of confidence levels in LLM responses might involve pre-training the model on synthetic data where knowledge points are cross-referenced and scored, or by incorporating intermediary thinking stages, similar to approaches in [5,6].

**Fast adaptation for personalization**\
A learning assistant should tailor materials to the learner's aptitude and personalize the learning experience based on their learning history. This history can be encoded into the assistant's behavior through RLHF, by prepending the history as a prefix context, or by storing it in an offline database for later RAG. However, the first two methods become uneconomical as the number of learners increases and the size of learning histories accumulates. The third method, while promising, requires rethinking as multi-modal histories, such as video and audio, become part of the assistant's slow memory. Therefore, new innovations are needed to enable fast adaptation of personalizable learning assistants, focusing on efficiently compressing learners' histories for personalized learning. Below are several promising research directions that could address these challenges and lead to a cost-efficient implementation of a personalizable learning assistant:

1. **Behavioral Adjustment through Joint Decoding:**\
    Conventional RLHF is not scalable for adapting LLMs to personalized learning, as it involves updating both a reward model and a policy model. A potential solution is to employ joint decoding at test time, where a reward model stores information that enables personalized assistance, similar to the approach in [7]. To facilitate seamless joint decoding with diverse reward models, an RLHF framework where the hyperparameters for joint decoding during rollouts are parameterized and learned would need to be developed.

2. **Efficient Compression of Learning History:**\
    As new modalities like video and audio become integral to personalized learning, storing uncompressed learning histories becomes impractical. One research direction could involve applying the principles of attention in LLMs as gradient updates to meta-parameters for selecting features of hidden activations, as suggested in [1]. This approach could enable the storage of billions of history tokens via meta-parameters, allowing the learning assistant to access distant parts of a learner's history. This method could complement the use of a neural retriever as an external database, as described below.

3. **Reformulating Retrieval with Neural Networks:**\
    Video and audio are essential for constructing adaptive responses to learner queries. However, storing and indexing this history using conventional databases becomes unscalable. A promising research direction would be to reformulate retrieval using an end-to-end neural network as a compressor for storing slow memory, as explored in [3], and to enhance the update of slow memory and retrieval efficiency using meta-knowledge, similar to the method proposed in [8].


## Reference
[1] Learning to (Learn at Test Time): RNNs with Expressive Hidden States, Y. Sun, preprint, 2024 \
[2] Physics of Language Models: Part 3.1, Knowledge Storage and Extraction, Z. Allen-Zhu and Y. Li, ICML 2024  \
[3] End-to-End Training of Neural Retrievers for Open-Domain Question Answering, D.S. Sachan et al., ACL 2021 \
[4] Towards Responsible Development of Generative AI for Education: An Evaluation-Driven Approach, I. Jurenka et al., preprint, 2024 \
[5] Thinking Tokens for Language Modeling, D. Herel and T. Mikolov, AITP, 2023 \
[6] Orca: Progressive Learning from Complex Explanation Traces of GPT-4, S. Mukherjee et al., preprint, 2023 \
[7] Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model, H. Deng and C. Raffel, ACL 2023 \
[8] Meta Knowledge for Retrieval Augmented Large Language Models, L. Mombaerts et al., KDD, 2024