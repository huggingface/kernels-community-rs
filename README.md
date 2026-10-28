# kernels-community-rs

> [!CAUTION]
> This repo is experimental and may contain breaking changes. Use with caution for now.


The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://hf.co/). You can find more information
about the Kernel Hub in the [docs](https://huggingface.co/docs/kernels/index).

This repository contains the sources for the Rust kernels in
[hf.co/kernels-community-rs](https://huggingface.co/kernels-community-rs). It is
the Rust counterpart of
[kernels-community](https://github.com/huggingface/kernels-community): CUDA
kernels are written in Rust and compiled with
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), host code is Rust, and
Python bindings go through tvm-ffi.

| Kernel | Backends | Hub |
| --- | --- | --- |
| [activation-rs](./activation-rs) | cuda | [kernels-community-rs/activation-rs](https://huggingface.co/kernels-community-rs/activation-rs) |

## Development

Builds are started by commenting on a PR (members and collaborators only):

```
/kernel-bot build <kernel>            # build + GPU tests, no upload
/kernel-bot build-and-stage <kernel>  # build, upload to kernels-staging-rs, GPU tests
/kernel-bot merge-and-upload <kernel> # merge the PR and release to kernels-community-rs
```
