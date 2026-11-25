{
  description = "stable-diffusion.cpp CUDA development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/b6018f87da91d19d0ab4cf979885689b469cdd41";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cuda = pkgs.cudaPackages_12_8;
      cudaToolkit = pkgs.symlinkJoin {
        name = "sd-cuda-toolkit";
        paths = with cuda; [
          cuda_nvcc
          cuda_cccl
          cuda_cudart
          libcublas.include
          libcublas.lib
        ];
      };
      mkShell = pkgs.mkShell.override { stdenv = cuda.backendStdenv; };
    in
    {
      devShells.${system}.default = mkShell {
        packages = with pkgs; [ cmake git ninja ];
        buildInputs = [ cudaToolkit ];
        CUDAToolkit_ROOT = "${cudaToolkit}";
        CUDACXX = "${cudaToolkit}/bin/nvcc";

        shellHook = ''
          if [ -r /usr/lib/wsl/lib/libcuda.so.1 ]; then
            export LD_LIBRARY_PATH="/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          fi
        '';
      };
    };
}
