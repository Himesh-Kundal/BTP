"""Inspection script printing full architectural table of SmolLM2-135M linear projections."""

from frontend.linear_extractor import extract_linear_modules
from frontend.model_loader import load_model_and_tokenizer


def main() -> None:
    print("Loading SmolLM2-135M at pinned revision...")
    model, _ = load_model_and_tokenizer()

    print("Extracting linear layers...")
    modules = extract_linear_modules(model)

    print("\n" + "=" * 105)
    print(f"{'Layer':<6} | {'Type':<10} | {'Shape [N, K]':<14} | {'Bias':<5} | {'Dtype':<8} | {'Module Path'}")
    print("=" * 105)

    # Print first layer (layer 0), middle layer (layer 14), and last layer (layer 29)
    sample_layers = {0, 14, 29}
    for m in modules:
        if m.layer_idx in sample_layers:
            shape_str = f"[{m.out_features}, {m.in_features}]"
            bias_str = "Yes" if m.has_bias else "No"
            print(f"{m.layer_idx:<6} | {m.proj_type:<10} | {shape_str:<14} | {bias_str:<5} | {m.dtype:<8} | {m.module_path}")
        elif m.layer_idx == 1 and m.proj_type == "q_proj":
            print("  ... [layers 1 to 13 omitted for display] ...")
        elif m.layer_idx == 15 and m.proj_type == "q_proj":
            print("  ... [layers 15 to 28 omitted for display] ...")

    print("=" * 105)
    total_params = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0) for m in modules)
    print(f"Total Linear Modules Extracted: {len(modules)} ({len(modules)//7} layers x 7 projection types)")
    print(f"Total Parameters in Linear Projections: {total_params:,} ({total_params / 1e6:.2f} M)")
    print("=" * 105)


if __name__ == "__main__":
    main()
