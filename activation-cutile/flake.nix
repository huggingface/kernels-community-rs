{
  description = "Flake for activation-cutile kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels/rust-cutile";
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
