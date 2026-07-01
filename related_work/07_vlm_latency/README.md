# VLM Latency

This bucket collects papers for the systems question: how can a VLM-based assistant reduce event-to-response latency without simply using a smaller model or a larger GPU?

## Visual Token Caching And Reuse

- `vlcache_visual_token_reuse_2512.12977.pdf` — reuses vision-token and KV-cache computations across similar multimodal inputs; the closest citation for cached visual tokens.
- `stc_cacher_streaming_token_compression_2512.00891.pdf` — caches ViT features for temporally similar frames and prunes streaming visual tokens before LLM prefill.

## Streaming Memory And Online Video

- `videollm_online_2406.11816.pdf` — streaming video-language inference/training baseline for online video understanding.
- `videostreaming_long_video_understanding_2405.16009.pdf` — separates streaming visual encoding from later reasoning over compact memories.
- `flash_vstream_2506.23825.pdf` — maintains short-term and long-term visual memory for real-time long-video understanding.
- `streamingvlm_infinite_streams_2510.09608.pdf` — keeps bounded streaming context for infinite video streams.
- `dispider_visual_streaming_assistant_2501.03218.pdf` — asynchronous visual-streaming assistant architecture.

## Token Reduction And Efficient VLMs

- `fastvlm_efficient_vision_encoder_2412.13303.pdf` — low-latency VLM through a faster vision encoder and fewer visual tokens.
- `llava_mini_one_vision_token_2501.03895.pdf` — compresses visual input aggressively before language-model reasoning.
- `evs_efficient_video_sampling_2510.14624.pdf` — reduces video token redundancy through efficient sampling.
- `streamingassistant_visual_token_pruning_2512.12560.pdf` — prunes visual tokens for online video understanding.
- `specvlm_speculative_decoding_2508.16201.pdf` — speculative decoding and video-token pruning for video LLMs.

## Serving And Runtime Systems

- `mminference_modality_aware_prefill_2504.16083.pdf` — modality-aware sparse prefill for long-context VLMs.
- `vllm_pagedattention_2309.06180.pdf` — KV-cache management and high-throughput LLM serving; useful as the serving baseline.
- `sglang_radixattention_2312.07104.pdf` — structured LM serving with cache reuse through RadixAttention; useful for repeated procedure/prompt programs.

## Relevance To This Project

The most project-relevant subset is `VLCache`, `STC-Cacher`, `VideoStreaming`, `Flash-VStream`, `MMInference`, `vLLM`, and `SGLang`. Together they support a procedure-aware runtime claim: known task structure can schedule visual evidence selection and pre-encoding before a reminder decision reaches the VLM.
