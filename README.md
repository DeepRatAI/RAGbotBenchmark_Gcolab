[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# RAGbotBenchmark_Gcolab: Digging into Open-Source RAG Bots on Colab GPUs! 🤖

Hey everyone! HermesIA (Gonzalo Romero) here. I got curious about how some of these popular open-source LLMs handle themselves in a practical Retrieval-Augmented Generation (RAG) setup. RAG is pretty standard now: you grab some relevant text based on a query, stuff it into the LLM's context along with the query, and let it generate an answer. Simple idea, but how do different models, especially on accessible hardware like Google Colab's T4 or A100 GPUs, *actually perform*? Does quantization *really* work well enough on a T4? How does a smaller model like Phi-2 stack up against LLaMA-2? 🤔

I wanted to move beyond just hearsay and get some **hands-on, empirical data**. Theory is one thing, but seeing how things run on actual hardware, measuring the latency, checking the output quality – that's where the real engineering insights are. My goal wasn't just to find the "best" model, but to **map out the behavior** of a few interesting setups and understand the **nitty-gritty trade-offs**.

## The Setup: What I Ran ⚙️

The core idea was straightforward:

1.  **Basic RAG:** Used SentenceTransformers (`all-MiniLM-L6-v2` seemed like a reasonable default) likely for embedding lookups to find relevant text snippets. Then, feed [Context + Query] into the LLM.
2.  **The Contenders:**
    * `microsoft/phi-2` on NVIDIA T4 (The small, fast one 🚀)
    * `meta-llama/Llama-2-7b-chat-hf` (4-bit quantized via `bitsandbytes`) on NVIDIA T4 (Big model squeezed onto free GPU 💪)
    * `meta-llama/Llama-2-7b-chat-hf` (Full Precision - BF16/FP16) on NVIDIA A100 (Big model on a powerful GPU ✨)
3.  **The Arena:** Google Colab. Represents a common development and experimentation environment. 💻
4.  **The Toolbox:** Standard stuff: Hugging Face (`transformers`, `accelerate`), `bitsandbytes` for quantization, `pytorch`, `SentenceTransformers`, `scikit-learn`, `umap-learn`, `pandas`.

## What I Looked For (Metrics & Analysis) 📊

For each setup, I ran N=15 diverse prompts and looked at:

* **Latency (Wall Clock Time):** Seriously, how long did `model.generate()` take in seconds? Is it fast enough for a chat? Measured carefully. ⏱️
* **Semantic Quality (BERTScore F1):** Do the generated answer embeddings (using `all-MiniLM-L6-v2` again) look similar to the reference answer embeddings? High score means similar meaning. (Scale 0-1, higher is better ✅).
* **Lexical Overlap (ROUGE-L F1):** Does the output *look* similar? Measures overlap in word sequences. Good for spotting fluency issues or verbatim copying. (Scale 0-1, higher is better ✅).
* **Output Embedding Geometry:** Where do the `all-MiniLM-L6-v2` embeddings of the *generated answers* live? Visualized with UMAP/t-SNE and poked at with KMeans. 🗺️

You can find all the code, plots, and detailed stats in the `Analysis_Report.ipynb` notebook – dive in! 🕵️‍♂️

## Key Findings / What Happened? (The Juicy Part!) 💡

Okay, here’s what jumped out from the data:

1.  **Speed: Hierarchy is Clear, Quantization Overhead is Stark!**
    * **Phi-2/T4:** Blazingly fast (~2.1s avg). Seriously impressive for its size.
    * **LLaMA-2/A100:** Quite fast too (~3.5s avg) and incredibly *consistent*. Rock solid timings, low variance. The A100 really helps here.
    * **The Surprise:** LLaMA-2 Quantized on T4 was *slow* (~10.1s avg, nearly 5x slower than Phi-2!) and more variable timings. Running those quantized ops on T4 seems to have significant overhead in this setup compared to the raw power of the A100 with the full model.
2.  **Quality - It's Complicated:**
    * **LLaMA-2/A100:** King of quality *and* consistency. Top scores on average for both BERTScore (meaning) and ROUGE-L (structure), and low variance. It delivers reliably. 👑
    * **Phi-2/T4:** The fascinating one. Its average BERTScore was almost identical to the mighty LLaMA-2/A100! It *understands* the semantics remarkably well. But, its ROUGE-L scores were all over the map (low median, high variance). It gets the idea right, but the phrasing varies a lot. 🤔
    * **LLaMA-2 Quant/T4:** Showed a measurable drop in average scores compared to its A100 counterpart on both metrics. It seems the aggressive 4-bit quantization here didn't just cost speed, it also shaved off some quality points. 📉
3.  **Embedding Space Tells a Story (UMAP/t-SNE):**
    * Plotting the answer embeddings was insightful. UMAP showed the LLaMA-2/A100 answers forming this tiny, dense ball – visually confirming its high consistency. Its outputs are semantically very similar to each other for these prompts.
    * Phi-2's answers were scattered widely across the plot – it explores a much larger semantic area, aligning with its variability in scores.
    * The LLaMA-2 Quant/T4 answers formed their *own* cluster, clearly separate from the A100 cluster. This suggests quantization doesn't just lower scores slightly, it actually shifts the *kinds* of embeddings the model produces. (Caveat: N=15 is tiny, so these are just hints!)
4.  **KMeans Clustered by... Topic!**
    * Tried KMeans (k=4) just to see what groups pop out. Interesting result: it mostly clustered answers based on the *input question topic*, not the model that generated them! This tells us more about the embedding space (MiniLM putting answers to same question close together) than about the models' styles, especially with only 15 data points.

## So What? / Takeaways ✅❌

* **No Free Lunch:** Getting good performance (speed+quality+consistency) often needs good hardware (A100). Quantization helps models *run* on less hardware, but can come with significant performance costs (especially latency!).
* **Horses for Courses:** Phi-2 is great for speed if you can handle variable output. LLaMA-2/A100 is the reliable workhorse if you have the GPU. LLaMA-2-Quant/T4 needs a use case that doesn't care much about speed (e.g., offline analysis).
* **Benchmark Empirically:** Don't assume performance based on model size or quantization level alone. **Measure it** on your target setup! Results might differ significantly with other quantization methods (GPTQ, AWQ?), different models, or even minor changes in software versions.
* **Look Beyond Metrics:** Visualizing embeddings (UMAP) and trying simple analyses (KMeans) gave extra clues about model behavior, even when KMeans gave an "unexpected" (but informative!) result. Don't just trust the average score! Dig deeper.

## Try It Yourself! Let's Tinker! 🛠️

This repo isn't just a report; it's a set of tools! Here’s how you can play with it:

1.  **Explore the Analysis:** Check out `Analysis_Report.ipynb` to see the results, plots, and detailed interpretations. Understand the baseline.
2.  **Run Your Own Benchmarks / Get More Data:** This is where it gets fun! The `benchmark_*.ipynb` files contain the actual RAG pipeline and evaluation code.
    * Pick one (e.g., `benchmark_phi2_T4.ipynb`).
    * Open it in Colab (make sure you select the **CORRECT GPU** - T4 for T4 notebooks, A100 for the A100 one!).
    * **<<<<< ⚠️ **CRITICAL STEP** --- **EDIT FILENAME BEFORE RUNNING** ⚠️ >>>>>** Find the cell that saves the results (look for `.to_csv(...)`). You **MUST** change the filename! If you don't, you'll overwrite the previous run's data. Call it something like `results/benchmark_results_phi2_RUN_02.csv`, then `_RUN_03.csv`, etc. **DO THIS. EVERY. SINGLE. TIME.** Seriously! 😉
    * Run the notebook. It will generate new answers and metrics, saving them to your *new* file.
    * Repeat for other configurations or more runs. More data = better analysis!
    * Go back to `Analysis_Report.ipynb`, update the data loading section (Cell 6) to include *all* your CSV files (original + new ones), and re-run the analysis on your bigger dataset! Does KMeans start finding model clusters with N=100? Does Phi-2's quality variance decrease? Let me know!
3.  **Adapt and Experiment:** Modify the prompts, swap retrieval methods (maybe try a sparse retriever?), try different embedding models (especially ones tuned for style?), benchmark other LLMs – fork it and go wild! Break things, fix them, learn!

## Nerding Out: Deeper Technical Observations & Speculation 🤓🔬

*(Okay, let's put on our thinking caps and speculate a bit more based on what we saw...)*

* **Quantization Latency Deep Dive:** That massive ~3x latency jump for LLaMA-2 4-bit on T4 vs. A100 (and nearly 5x vs Phi-2!) demands attention. While `bitsandbytes` enables 4-bit computation, the T4's architecture (pre-Hopper) likely lacks highly optimized kernels for these low-bit operations compared to A100's optimized FP16/BF16 Tensor Cores. My educated guess: the T4 might be spending considerable time in software routines for de-quantizing weights to FP16/BF16 for the actual matrix multiplications, or facing memory bandwidth bottlenecks shuffling these converted weights around, rather than executing efficient, native low-bit ops. The A100, conversely, flies through FP16/BF16. This empirical result is a strong reminder: quantization's *practical* efficiency is deeply tied to hardware-specific kernel support. Profiling GPU utilization and kernel timings would be needed to confirm the exact bottleneck.
* **Embedding Space Geometry & Representational Shifts:** The UMAP plots were gold. The tight LLaMA-2/A100 cluster visually represents its low variance in quality metrics – it consistently maps inputs to a stable region of the semantic manifold. Phi-2's wide scatter reflects its higher output variance. The most intriguing part is the *distinct spatial location* of the LLaMA-2 Quant/T4 cluster relative to the A100 one. This isn't just noise; it suggests a **systematic representational shift** induced by 4-bit quantization in this setup. The model isn't just slightly less accurate; it represents concepts or styles *differently*, occupying a different neighborhood in the `all-MiniLM-L6-v2` embedding space. What does this *mean* for downstream tasks? Does it develop biases? Fascinating question needing more study (and data!).
* **Why KMeans Found Topics (Feature Engineering Matters!):** The KMeans outcome is a perfect illustration of feature dominance. `all-MiniLM-L6-v2` is optimized to make topic similarity the primary axis of variation captured by L2 distance. It does its job well! Answers about "gradient descent" are pulled together strongly. Any model-specific stylistic signals encoded in the MiniLM embeddings are likely much weaker *in terms of L2 distance*, especially with only N=15 points to work with. So, KMeans naturally partitions along the high-variance topic axis. It validates the embedding's semantic relevance but shows its limitations for *this specific task* of unsupervised style clustering with KMeans+L2. A different embedding model designed for style, or a different clustering algorithm/distance metric, might be needed to surface those subtler model differences.
* **Phi-2: Semantic Powerhouse, Structural Wildcard?** The BERTScore/ROUGE-L divergence is classic. High BERTScore suggests the core semantic concepts are being captured effectively – impressive for its size! The low/variable ROUGE-L suggests it uses diverse phrasing and sentence structures to express that core meaning, unlike LLaMA-2/A100 which seems more "canonical" in its output structure (leading to higher ROUGE-L *and* BERTScore consistency). It implies Phi-2 might be good for tasks needing semantic understanding but less suitable where precise phrasing or structural mimicry is key, unless prompted carefully.
* **A100 Consistency: The Virtuous Cycle?** The "rock-solid" performance of LLaMA-2/A100 (latency, metrics, UMAP) is likely a synergy effect. The raw compute power enables stable, fast execution using higher precision (FP16/BF16). This numerical stability might lead to more consistent internal activation patterns during generation, resulting in outputs that are not only high quality but also reliably land within a tight region of the semantic embedding space. It's a well-behaved system operating comfortably within its limits.

This benchmark, though small-scale, generated rich data precisely because we looked at it from multiple angles. The path forward involves scaling up N and perhaps exploring different embedding spaces!

## Files 📁

```
Benchmark_ChatbotRAG/
├── Analysis_Report.ipynb      # <--- The main analysis is here!
├── results/                   # <--- Benchmark output CSVs go here
│   ├── benchmark_results_phi2.csv
│   ├── benchmark_results_llama2_quant_t4.csv
│   └── benchmark_results_llama2_chat_a100.csv
│   └── (... add your own generated files here!)
├── benchmark_phi2_T4.ipynb           # <--- Run this for Phi-2/T4 benchmarks
├── benchmark_llama2_quant_T4.ipynb  # <--- Run this for LLaMA-2 Quant/T4
├── benchmark_llama2_fp_A100.ipynb    # <--- Run this for LLaMA-2 Full/A100
├── images/                     # (Optional place for saved images)
│   └── ...
├── .gitignore                  # Tells Git what to ignore
└── README.md                   # This file you're reading!
```

## Setup 🔧

Pretty standard Python stuff:
1.  `git clone ...`
2.  `cd Benchmark_ChatbotRAG`
3.  `pip install -r requirements.txt` (You might need to create this file based on the imports in the notebooks: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `sentence-transformers`, `torch`, `umap-learn`, `transformers`, `accelerate`, `bitsandbytes`, `ipykernel`)
4.  Run the notebooks!

Hope you find this useful for understanding the practical side of running these RAG systems. Have fun digging in and let me know if you find cool stuff!
---
