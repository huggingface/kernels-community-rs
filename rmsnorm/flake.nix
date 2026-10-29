{
  description = "Flake for rmsnorm kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/rust-gpu-support-2";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
