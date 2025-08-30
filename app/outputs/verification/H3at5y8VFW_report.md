# Claim Verification Report

## Summary

- Total Claims: 42
- Supported: 16 (38.1%)
- Partially Supported: 19 (45.2%)
- Contradicted: 0 (0.0%)
- Undetermined: 7 (16.7%)

## Detailed Results

### Claim 1

**Claim:** Summary: This paper introduces Self-Retrieval, an end-to-end IR system driven entirely by a single LLM.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about Self-Retrieval being an end-to-end IR system driven by a single LLM.

**Evidence:**
1. Recently, information retrieval (IR) systems and large language models (LLMs) have witnessed a growing synergy, with advancements in one field driving progress in the other [13, 56]. On one hand, IR s...
2. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
3. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
4. LLMfor IR Recent studies have explored leveraging LLMs to enhance various components of IR systems, including query rewriting, retrieval, and reranking. For query rewriting, LLMs have been employed to...

--------------------------------------------------

### Claim 2

**Claim:** The authors provide experimental evidence showing that Self-Retrieval outperforms traditional sparse, dense, and generative retrieval methods on benchmark datasets like NQ and TriviaQA.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about Self-Retrieval outperforming methods on NQ and TriviaQA.

**Evidence:**
1. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
2. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...

--------------------------------------------------

### Claim 3

**Claim:** The integration of all IR functions into a single LLM is a novel contribution that leverages the inherent capabilities of LLMs across the full spectrum of IR tasks, offering a streamlined and potentially more effective approach.

**Result:** Supported
**Confidence:** 0.95
**Justification:** Evidence shows Self-Retrieval integrates IR functions into a single LLM, enhancing performance and streamlining processes.

**Evidence:**
1. Recently, information retrieval (IR) systems and large language models (LLMs) have witnessed a growing synergy, with advancements in one field driving progress in the other [13, 56]. On one hand, IR s...
2. LLMfor IR Recent studies have explored leveraging LLMs to enhance various components of IR systems, including query rewriting, retrieval, and reranking. For query rewriting, LLMs have been employed to...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. 1

## Self-Retrieval: End-to-End Information Retrieval with One Large Language Model

Qiaoyu Tang 1 , 2 ∗ , Jiawei Chen 1 , 2 ∗ , Zhuoqun Li 1 , 2 , Bowen Yu 3 , Yaojie Lu 1 , Cheng Fu 3 , Haiyang Yu ...

--------------------------------------------------

### Claim 4

**Claim:** The concept of Self-Retrieval is introduced clearly, making it accessible to readers.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence explains Self-Retrieval's technical aspects but does not explicitly state it is introduced clearly or made accessible to readers.

**Evidence:**
1. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
2. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
3. In this section, we introduce our proposed Self-Retrieval. The overall architecture is illustrated in . Different from traditional information retrieval systems that separate indexing, retrieval, and ...
4. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...

--------------------------------------------------

### Claim 5

**Claim:** The paper presents good experimental results that demonstrate significant improvements over existing retrieval methods.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence shows significant improvements in experimental results over existing retrieval methods.

**Evidence:**
1. Datasets and metrics We conduct main experiments on Natural Questions (NQ) [21] and TriviaQA [18] datasets, both of which are widely used retrieval benchmarks based on Wikipedia. We use their versions...
2. from Wikipedia for each dataset. Each document is segmented into passages of maximum 200 words, yielding approximately 1 million passages in total. The detailed statistics of the datasets are presente...
3. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...
4. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...

--------------------------------------------------

### Claim 6

**Claim:** Weaknesses: 1.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence discusses weaknesses of IR systems but does not directly address the specific claim about 'Weaknesses: 1.'

**Evidence:**
1. Recently, information retrieval (IR) systems and large language models (LLMs) have witnessed a growing synergy, with advancements in one field driving progress in the other [13, 56]. On one hand, IR s...
2. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...
3. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
4. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...

--------------------------------------------------

### Claim 7

**Claim:** - It is better to include more experimental results on the full KILT datasets as many existing studies for a fair comparison.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The claim suggests more experimental results on full KILT datasets are needed, but the evidence only provides results on NQ and TriviaQA subsets.

**Evidence:**
1. Datasets and metrics We conduct main experiments on Natural Questions (NQ) [21] and TriviaQA [18] datasets, both of which are widely used retrieval benchmarks based on Wikipedia. We use their versions...
2. ## A Dataset Statistics

Table 6 presents the statistics of the NQ and TriviaQA datasets used in our experiments.

| Dataset   | Natural Questions   | Natural Questions   | TriviaQA   | TriviaQA   |
|...
3. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...
4. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...

--------------------------------------------------

### Claim 8

**Claim:** Ensuring consistent use of key terms throughout the paper would improve its readability and professionalism.

**Result:** Undetermined
**Confidence:** 0.95
**Justification:** The evidence does not address the claim about key term consistency affecting readability and professionalism.

**Evidence:**
1. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...
2. | [19]   | Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webbe...
3. During training, we utilize the gold passage from the supervision data as the positive instance, while sampling negative instances from both the same and different documents. This training strategy co...
4. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference...

--------------------------------------------------

### Claim 9

**Claim:** Questions: Please see my concerns in weaknesses.

**Result:** Undetermined
**Confidence:** 0.80
**Justification:** The claim references 'weaknesses' but no specific weaknesses are discussed in the evidence.

**Evidence:**
1. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...
2. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
3. |          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
|          | H@1   | H@5   | M@5   | H@1        | H@5 ...
4. ## A Dataset Statistics

Table 6 presents the statistics of the NQ and TriviaQA datasets used in our experiments.

| Dataset   | Natural Questions   | Natural Questions   | TriviaQA   | TriviaQA   |
|...

--------------------------------------------------

### Claim 10

**Claim:** Limitations: n/a

Soundness: 3

Presentation: 3

Contribution: 3

Ethics Review Flagged: ['No ethics review needed.']

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence shows performance degradation without certain components, but does not directly confirm the claim's specific metrics or context.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. from Wikipedia for each dataset. Each document is segmented into passages of maximum 200 words, yielding approximately 1 million passages in total. The detailed statistics of the datasets are presente...
3. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
4. - [1] Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, Shengling Gao, J. Guo, Xiangnan He, Yanyan Lan, Chenliang Li, Yiqun Liu,...

--------------------------------------------------

### Claim 11

**Claim:** Summary: This paper proposes Self-Retrieval, an LM that retrieve, rerank passages, and generate answers using a single model.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about Self-Retrieval unifying retrieval tasks into a single model.

**Evidence:**
1. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
2. In this section, we introduce our proposed Self-Retrieval. The overall architecture is illustrated in . Different from traditional information retrieval systems that separate indexing, retrieval, and ...
3. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...
4. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...

--------------------------------------------------

### Claim 12

**Claim:** Evaluation shows that Self-Retrieval outperforms previous dense and generative retrievers in retrieval tasks and achieves better EM scores in answer generation tasks.

**Result:** Supported
**Confidence:** 0.95
**Justification:** Evidence shows Self-Retrieval outperforms generative retrievers in MRR@5 and EM scores.

**Evidence:**
1. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
2. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
3. Generative retrieval Generative retrieval methods leverage sequence-to-sequence language models to generate document identifiers for a given query [8, 42]. This paradigm is pioneered by GENRE [7], whi...
4. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...

--------------------------------------------------

### Claim 13

**Claim:** The concept is novel.

**Result:** Undetermined
**Confidence:** 0.95
**Justification:** The evidence does not directly address the novelty of the concept.

**Evidence:**
1. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...
2. | [33]   | Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. From doc2query to doctttttquery. Online preprint , 6:2, 2019. |
|--------|----------|
| [34]   | Fabio Petroni, Aleksandra Piktus, Angela Fan,...
3. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
4. |      | Emad Mostaque, Michael Pieler, Nikhil Pinnaparju, Paulo Rocha, Harry Saini, Hannah Teufel, Niccolo Zanichelli, and Carlos Riquelme. Stable lm 2 1.6b technical report, 2024. |
|------|--------...

--------------------------------------------------

### Claim 14

**Claim:** Self-Retrieval achieves the best performance for both passage retrieval and answer generation tasks.

**Result:** Supported
**Confidence:** 0.95
**Justification:** Evidence shows Self-Retrieval outperforms other generative methods in passage retrieval and answer generation.

**Evidence:**
1. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
2. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
3. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
4. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...

--------------------------------------------------

### Claim 15

**Claim:** Self-Retrieval shows promising results when the corpus size scales to 3 million.

**Result:** Supported
**Confidence:** 0.95
**Justification:** Evidence shows Self-Retrieval maintains performance degradation comparable to BGE-FT when scaling to 3 million passages.

**Evidence:**
1. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
2. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...
3. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
4. In this section, we introduce our proposed Self-Retrieval. The overall architecture is illustrated in . Different from traditional information retrieval systems that separate indexing, retrieval, and ...

--------------------------------------------------

### Claim 16

**Claim:** Weaknesses: 1.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence discusses weaknesses of IR systems but does not directly address the specific claim about 'Weaknesses: 1.'

**Evidence:**
1. Recently, information retrieval (IR) systems and large language models (LLMs) have witnessed a growing synergy, with advancements in one field driving progress in the other [13, 56]. On one hand, IR s...
2. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...
3. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
4. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...

--------------------------------------------------

### Claim 17

**Claim:** The proposed model includes an in-domain fine-tuned reranker, while the baseline BGE-FT + reader does not have a reranking stage.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms Self-Retrieval includes a reranker while BGE-FT + reader lacks it.

**Evidence:**
1. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...
2. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
3. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference...
4. Baselines We evaluate Self-Retrieval models for both passage retrieval and document retrieval, comparing them with sparse, dense, and generative retrieval baselines. The sparse retrieval baselines are...

--------------------------------------------------

### Claim 18

**Claim:** This may make the comparison unfair since reranking can significantly improve RAG results.

**Result:** Supported
**Confidence:** 0.95
**Justification:** Evidence shows reranking improves RAG results, supporting the claim about unfair comparisons.

**Evidence:**
1. |          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
|          | H@1   | H@5   | M@5   | H@1        | H@5 ...
2. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...

--------------------------------------------------

### Claim 19

**Claim:** Compare the efficiency and effectiveness of using shared models versus separate models for the three steps in Self-Retrieval?

**Result:** Undetermined
**Confidence:** 0.95
**Justification:** The evidence discusses Self-Retrieval's performance but does not compare shared vs separate models for the three steps.

**Evidence:**
1. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
2. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
3. Recently, information retrieval (IR) systems and large language models (LLMs) have witnessed a growing synergy, with advancements in one field driving progress in the other [13, 56]. On one hand, IR s...
4. 1

## Self-Retrieval: End-to-End Information Retrieval with One Large Language Model

Qiaoyu Tang 1 , 2 ∗ , Jiawei Chen 1 , 2 ∗ , Zhuoqun Li 1 , 2 , Bowen Yu 3 , Yaojie Lu 1 , Cheng Fu 3 , Haiyang Yu ...

--------------------------------------------------

### Claim 20

**Claim:** Limitations: The limitations are adequately discussed.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence discusses limitations but does not fully confirm the claim's adequacy.

**Evidence:**
1. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
2. - [1] Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, Shengling Gao, J. Guo, Xiangnan He, Yanyan Lan, Chenliang Li, Yiqun Liu,...
3. | [33]   | Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. From doc2query to doctttttquery. Online preprint , 6:2, 2019. |
|--------|----------|
| [34]   | Fabio Petroni, Aleksandra Piktus, Angela Fan,...
4. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...

--------------------------------------------------

### Claim 21

**Claim:** Soundness: 3

Presentation: 3

Contribution: 3

Ethics Review Flagged: ['No ethics review needed.']

**Result:** Undetermined
**Confidence:** 0.80
**Justification:** The evidence does not directly address the claim's metrics or their scores.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. Baselines We evaluate Self-Retrieval models for both passage retrieval and document retrieval, comparing them with sparse, dense, and generative retrieval baselines. The sparse retrieval baselines are...
3. During training, we utilize the gold passage from the supervision data as the positive instance, while sampling negative instances from both the same and different documents. This training strategy co...
4. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...

--------------------------------------------------

### Claim 22

**Claim:** Summary: This paper introduces Self-Retrieval, a new generative retrieval architecture.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about Self-Retrieval's effectiveness and architecture.

**Evidence:**
1. We evaluate Self-Retrieval on three representative retrieval benchmarks: NQ, TriviaQA, and MS MARCO. Experimental results demonstrate that Self-Retrieval substantially outperforms existing sparse retr...
2. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
3. Generative retrieval Generative retrieval methods leverage sequence-to-sequence language models to generate document identifiers for a given query [8, 42]. This paradigm is pioneered by GENRE [7], whi...
4. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...

--------------------------------------------------

### Claim 23

**Claim:** On subset of NQ and TriviaQA, this approach significantly outperforms existing dual encoders and generative retrieval models.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence shows Self-Retrieval outperforms existing models on NQ and TriviaQA subsets.

**Evidence:**
1. Generative retrieval Generative retrieval methods leverage sequence-to-sequence language models to generate document identifiers for a given query [8, 42]. This paradigm is pioneered by GENRE [7], whi...
2. Datasets and metrics We conduct main experiments on Natural Questions (NQ) [21] and TriviaQA [18] datasets, both of which are widely used retrieval benchmarks based on Wikipedia. We use their versions...
3. |          | NQ    | NQ    | TriviaQA   | TriviaQA   |
|----------|-------|-------|----------|----------|
|          | 10K   | 40K   | 10K        | 40K        |
| BGE-FT + StableLM-FT | 43.18 | 41.24 ...
4. LLMfor IR Recent studies have explored leveraging LLMs to enhance various components of IR systems, including query rewriting, retrieval, and reranking. For query rewriting, LLMs have been employed to...

--------------------------------------------------

### Claim 24

**Claim:** The paper presents a self-supervise object to help model memorize the corpus.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence confirms the paper describes a self-supervised method for corpus memorization via internalizing data into LLM parameters.

**Evidence:**
1. Self-Retrieval integrates indexing into the LLM's parameters through self-supervised learning, enabling the model to internalize the entire corpus. Unlike generative retrieval methods that rely on com...
2. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
3. In this section, we introduce our proposed Self-Retrieval. The overall architecture is illustrated in . Different from traditional information retrieval systems that separate indexing, retrieval, and ...
4. In this paper, we introduce Self-Retrieval, an end-to-end information retrieval architecture driven entirely by one large language model. This integration is not trivial due to the inherent mismatch b...

--------------------------------------------------

### Claim 25

**Claim:** - Strong quality improvements.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** Evidence shows Self-Retrieval has quality improvements over some methods, but not all aspects of the claim are explicitly confirmed.

**Evidence:**
1. We comprehensively evaluate Self-Retrieval against various two-stage retriever-reranker pipelines. Specifically, we construct these pipelines using state-of-the-art retrievers (BGE, GTR, GritLM, and D...
2. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
3. |          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
|          | H@1   | H@5   | M@5   | H@1        | H@5 ...
4. During training, we utilize the gold passage from the supervision data as the positive instance, while sampling negative instances from both the same and different documents. This training strategy co...

--------------------------------------------------

### Claim 26

**Claim:** The paper reports substantial improvements over previous dual-encoder approaches and generative retrieval models.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence shows Self-Retrieval outperforms generative retrieval models and dual-encoder approaches in passage retrieval.

**Evidence:**
1. Generative retrieval Generative retrieval methods leverage sequence-to-sequence language models to generate document identifiers for a given query [8, 42]. This paradigm is pioneered by GENRE [7], whi...
2. - DPR [19] is a dual-encoder model trained with in-batch negative sampling. We fine-tune DPR on our training datasets to obtain DPR-FT , following the official implementation and hyperparameter settin...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. Baselines We evaluate Self-Retrieval models for both passage retrieval and document retrieval, comparing them with sparse, dense, and generative retrieval baselines. The sparse retrieval baselines are...

--------------------------------------------------

### Claim 27

**Claim:** - Ablations shows that the model scales well with model size.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** Ablation study shows performance degradation, but no direct evidence of scaling with model size.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...
3. - Ultron [55] represents documents using three types of identifiers (URL, PQ, Atomic) and trains the model through a progressive three-stage pipeline.
- DynamicRetriever [54] parameterizes traditional...
4. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...

--------------------------------------------------

### Claim 28

**Claim:** Previous dense retrievers often plateau due to the bottleneck layer; the scaling curve of this new architecture is promising.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence discusses scalability but does not directly address dense retrievers plateauing or the new architecture's scaling curve.

**Evidence:**
1. We comprehensively evaluate Self-Retrieval against various two-stage retriever-reranker pipelines. Specifically, we construct these pipelines using state-of-the-art retrievers (BGE, GTR, GritLM, and D...
2. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...
3. Scaling corpus size Recent studies [35, 53] have demonstrated that generative retrieval methods such as DSI or NCI experience more significant performance degradation compared to dense retrieval metho...
4. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...

--------------------------------------------------

### Claim 29

**Claim:** Weaknesses: - Experiment only used wikipedia-based datasets.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The claim mentions using Wikipedia-based datasets, which is supported by the evidence, but lacks specific details on dataset limitations.

**Evidence:**
1. Datasets and metrics We conduct main experiments on Natural Questions (NQ) [21] and TriviaQA [18] datasets, both of which are widely used retrieval benchmarks based on Wikipedia. We use their versions...
2. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...
3. from Wikipedia for each dataset. Each document is segmented into passages of maximum 200 words, yielding approximately 1 million passages in total. The detailed statistics of the datasets are presente...
4. ## A Dataset Statistics

Table 6 presents the statistics of the NQ and TriviaQA datasets used in our experiments.

| Dataset   | Natural Questions   | Natural Questions   | TriviaQA   | TriviaQA   |
|...

--------------------------------------------------

### Claim 30

**Claim:** However, wikipedia is heavily used in pretraining, so it is unclear if the proposed approach can let model sufficiently memorize other datasets.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence mentions Wikipedia use in pretraining but does not confirm if the approach ensures sufficient memorization of other datasets.

**Evidence:**
1. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...
2. Datasets and metrics We conduct main experiments on Natural Questions (NQ) [21] and TriviaQA [18] datasets, both of which are widely used retrieval benchmarks based on Wikipedia. We use their versions...
3. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
4. Self-Retrieval integrates indexing into the LLM's parameters through self-supervised learning, enabling the model to internalize the entire corpus. Unlike generative retrieval methods that rely on com...

--------------------------------------------------

### Claim 31

**Claim:** - Missing dense retrieval + cross-attention reranking baselines.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** Evidence mentions dense retrieval baselines but not cross-attention reranking specifically.

**Evidence:**
1. Baselines We evaluate Self-Retrieval models for both passage retrieval and document retrieval, comparing them with sparse, dense, and generative retrieval baselines. The sparse retrieval baselines are...
2. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference...

--------------------------------------------------

### Claim 32

**Claim:** Since the proposed method's reranking stage essentially uses cross attention to judge the query and the retrieved candidate passage, the computational cost of the reranking stage is similar to that of a separate cross-attention reranker.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence discusses computational efficiency but does not directly compare reranking costs to a separate cross-attention reranker.

**Evidence:**
1. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...
2. LLMfor IR Recent studies have explored leveraging LLMs to enhance various components of IR systems, including query rewriting, retrieval, and reranking. For query rewriting, LLMs have been employed to...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference...

--------------------------------------------------

### Claim 33

**Claim:** The ablation in Table 3 seems to show that the proposed method's retrieval-alone performance is stronger than most retrieval baselines, but the paper can be more convincing if having e2e comparison to other 2-stage retrieval pipelines like BGE + BGE reranker or GTR + RankT5.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence supports retrieval performance but lacks explicit mention of e2e comparisons to specific 2-stage pipelines.

**Evidence:**
1. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
2. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
3. Baselines We evaluate Self-Retrieval models for both passage retrieval and document retrieval, comparing them with sparse, dense, and generative retrieval baselines. The sparse retrieval baselines are...
4. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...

--------------------------------------------------

### Claim 34

**Claim:** - Lacking efficiency discussion.

**Result:** Undetermined
**Confidence:** 0.95
**Justification:** The evidence does not mention efficiency discussion in the claim.

**Evidence:**
1. |          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
|          | H@1   | H@5   | M@5   | H@1        | H@5 ...
2. We comprehensively evaluate Self-Retrieval against various two-stage retriever-reranker pipelines. Specifically, we construct these pipelines using state-of-the-art retrievers (BGE, GTR, GritLM, and D...
3. | [33]   | Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. From doc2query to doctttttquery. Online preprint , 6:2, 2019. |
|--------|----------|
| [34]   | Fabio Petroni, Aleksandra Piktus, Angela Fan,...
4. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...

--------------------------------------------------

### Claim 35

**Claim:** Limitations: NA

Soundness: 3

Presentation: 4

Contribution: 3

Ethics Review Flagged: ['No ethics review needed.']

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence shows ablation results but does not directly support the specific scores or metrics mentioned in the claim.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. from Wikipedia for each dataset. Each document is segmented into passages of maximum 200 words, yielding approximately 1 million passages in total. The detailed statistics of the datasets are presente...
3. - [1] Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, Shengling Gao, J. Guo, Xiangnan He, Yanyan Lan, Chenliang Li, Yiqun Liu,...
4. Baselines We evaluate Self-Retrieval models for both passage retrieval and document retrieval, comparing them with sparse, dense, and generative retrieval baselines. The sparse retrieval baselines are...

--------------------------------------------------

### Claim 36

**Claim:** Summary: The paper proposes an approach of self-retrieval, which uses the probability of generation of the passage as the ranking criterion.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The evidence directly supports the claim about Self-Retrieval using generation probability for ranking.

**Evidence:**
1. In this section, we introduce our proposed Self-Retrieval. The overall architecture is illustrated in . Different from traditional information retrieval systems that separate indexing, retrieval, and ...
2. Training Self-Retrieval unifies the three distinct tasks of information retrieval - indexing, retrieval, and reranking - into text generation tasks, trained using cross-entropy loss in an auto-regress...
3. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
4. Analysis on reranking In this part, we conduct an in-depth analysis of the reranking performance of Self-Retrieval reranker module in comparison with the fine-tuned BGE-Reranker. We employ DPR-FT, SEA...

--------------------------------------------------

### Claim 37

**Claim:** The proposed method outperforms the others.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence shows Self-Retrieval performs well but does not directly compare all methods in the claim.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pages 9844-9855, Abu Dhabi, United Arab Emirates, ...
3. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...
4. | Method          |   R@1 | R@5   |   M@10 |
|----------|-------|-------|--------|
| Sparse Retrieval BM25 [37]          |  18.9 | 42.8  |   29.2 |
| DocT5Query [28]          |  23.3 | 49.4  |   34.8 ...

--------------------------------------------------

### Claim 38

**Claim:** The proposed method thus has some novelty compared to the literature.

**Result:** Supported
**Confidence:** 0.95
**Justification:** The claim is supported by evidence showing Self-Retrieval outperforms existing methods, indicating novelty.

**Evidence:**
1. | Method          |   R@1 | R@5   |   M@10 |
|----------|-------|-------|--------|
| Sparse Retrieval BM25 [37]          |  18.9 | 42.8  |   29.2 |
| DocT5Query [28]          |  23.3 | 49.4  |   34.8 ...
2. | [19]   | Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webbe...
3. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
4. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...

--------------------------------------------------

### Claim 39

**Claim:** The experimental results are very good, showing improved performance on document retrieval and QA.

**Result:** Partially Supported
**Confidence:** 0.85
**Justification:** The evidence mentions improved performance on retrieval metrics but does not explicitly state 'very good' or QA improvements.

**Evidence:**
1. Datasets and metrics We conduct main experiments on Natural Questions (NQ) [21] and TriviaQA [18] datasets, both of which are widely used retrieval benchmarks based on Wikipedia. We use their versions...
2. from Wikipedia for each dataset. Each document is segmented into passages of maximum 200 words, yielding approximately 1 million passages in total. The detailed statistics of the datasets are presente...
3. While our experiments demonstrate the effectiveness of Self-Retrieval, several limitations need to be addressed in future work. Our current evaluation is limited to 200K Wikipedia documents and 3M pas...
4. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference...

--------------------------------------------------

### Claim 40

**Claim:** Weaknesses: A key idea is the use of trie for passage generation.

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence mentions trie usage for constrained decoding but does not directly link it to passage generation weaknesses.

**Evidence:**
1. However, since LLMs are general-purpose pre-trained models rather than statistical frequency models, the generated passage ˆ p may not exactly match any passage in D , making it challenging to locate ...
2. Our results indicate that other generative retrieval baselines exhibit suboptimal performance on passage retrieval. Even the largest DSI-XXL model only achieves an MRR@5 of 50.20 on NQ, significantly ...
3. |
| [13] | Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey, 202...
4. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference...

--------------------------------------------------

### Claim 41

**Claim:** More details should be presented.

**Result:** Undetermined
**Confidence:** 0.80
**Justification:** The claim lacks specific details to match evidence about performance or components.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. |          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
|          | H@1   | H@5   | M@5   | H@1        | H@5 ...
3. | Method          |   R@1 | R@5   |   M@10 |
|----------|-------|-------|--------|
| Sparse Retrieval BM25 [37]          |  18.9 | 42.8  |   29.2 |
| DocT5Query [28]          |  23.3 | 49.4  |   34.8 ...
4. |          |        | NQ      | NQ    | NQ      | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|--------|---------|-------|---------|----------|----------|----------|
| Model          | Params | ...

--------------------------------------------------

### Claim 42

**Claim:** Limitations: yes

Soundness: 3

Presentation: 3

Contribution: 3

Ethics Review Flagged: ['No ethics review needed.']

**Result:** Partially Supported
**Confidence:** 0.75
**Justification:** The evidence shows ablation results but does not directly confirm the claim's specific metrics or ratings.

**Evidence:**
1. Table 4: Ablation study on NQ and TriviaQA.

| Method          | NQ    | NQ    | NQ    | TriviaQA   | TriviaQA   | TriviaQA   |
|----------|-------|-------|-------|----------|----------|----------|
| ...
2. - [1] Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, Shengling Gao, J. Guo, Xiangnan He, Yanyan Lan, Chenliang Li, Yiqun Liu,...
3. from Wikipedia for each dataset. Each document is segmented into passages of maximum 200 words, yielding approximately 1 million passages in total. The detailed statistics of the datasets are presente...
4. During training, we utilize the gold passage from the supervision data as the positive instance, while sampling negative instances from both the same and different documents. This training strategy co...

--------------------------------------------------

