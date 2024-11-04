
# MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs.

Data and code for our paper [An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs](https://arxiv.org/abs/2410.04698).

For more details, please refer to the project page: https://mathhay.github.io/.

[[Webpage](https://mathhay.github.io/)] [[Paper](https://arxiv.org/abs/2410.04698)] [[Huggingface Dataset]()] [[Leaderboard]()] [[Twitter]()]

<!-- :star: Our data and method have inspired or been used for the development of recent large language models (LLMs) including [Google's Gemini](https://gemini.google.com), [Perplexity.AI's Online LLMs](https://blog.perplexity.ai/blog/introducing-pplx-online-llms), [You.com](https://about.you.com/introducing-the-you-api-web-scale-search-for-llms), and [Contextual AI's RAG 2.0](https://contextual.ai/introducing-rag2) :star: -->


## 💥 News 💥
- **[2024.11.14]** Our code is now accessible.
- **[2024.10.07]** Our paper is now accessible at https://arxiv.org/abs/2410.04698.


### Overview of the automatic construction of MATHHAY:
<p align="center">
    <img src="assets/framework.png" width="80%"> <br>
  Overview of the framework for the automatic construction of the <b>MATHHAY</b> Benchmark.
</p>

### Compared to existing long-context benchmarks:

| Benchmark            | Multi-Doc | Multi-Step | Avoid Contam. | Irrelevant Docs | Realistic Docs | Auto. Const. | Math. Reasoning |
|----------------------|-----------|------------|---------------|-----------------|----------------|--------------|-----------------|
| ZeroSCROLLS          | ✓         | ✓          | ✗             | ✓               | ✓              | ✗            | ✗               |
| L-Eval (Math)        | ✓         | ✓          | ✓             | ✗               | ✗              | ✓            | ✓               |
| LongBench            | ✓         | ✓          | ✓             | ✓               | ✓              | ✗            | ✗               |
| BAMBOO               | ✓         | ✗          | ✓             | ✓               | ✓              | ✗            | ✗               |
| InfiniteBench (Math) | ✓         | ✓          | ✗             | ✓               | ✗              | ✗            | ✓               |
| Loong                | ✓         | ✓          | ✓             | ✓               | ✓              | ✗            | ✗               |
| NIAH                 | ✗         | ✗          | ✗             | ✓               | ✓              | ✓            | ✗               |
| RULER                | ✓         | ✓          | ✗             | ✓               | ✓              | ✓            | ✗               |
| FlenQA               | ✗         | ✓          | ✗             | ✓               | ✓              | ✗            | ✗               |
| SummHay              | ✓         | ✗          | ✓             | ✓               | ✓              | ✗            | ✗               |
| BABILong             | ✓         | ✓          | ✓             | ✓               | ✓              | ✗            | ✗               |
| NeedleBench          | ✓         | ✓          | ✗             | ✓               | ✓              | ✓            | ✗               |
| **MathHay (Ours)**   | ✓         | ✓          | ✓             | ✓               | ✓              | ✓            | ✓               |



### Leaderboard on the MathHay V1:

Accuracy scores on the **MathHay** V1:


python -m spacy download en_core_web_sm


## Citation


If you use our data or method, please cite our paper:
```bibtex
@article{wang2024mathhay,
  title={MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs},
  author={Wang, Lei and Dong, Shan and Xu, Yuhui and Dong, Hanze and Wang, Yalu and Saha, Amrita and Lim, Ee-Peng and Xiong, Caiming and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2410.04698},
  year={2024}
}
```

