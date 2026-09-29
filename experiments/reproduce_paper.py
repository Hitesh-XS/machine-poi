#!/usr/bin/env python3
"""
Demonstration runner for the Machine-POI steering library.

It prints sample outputs; it is not the evidence for PAPER.md. Steering claims
cite experiments/steering_eval.py, which scores held-out prompts with
confidence intervals. Sections:
1. 5.1: steered and unsteered outputs for a few prompts
2. mra: one multi-resolution retrieval answer
3. 5.3: outputs across dose ratios

Section 5.2, which counted English keywords as a "thematic" score, is retired;
use the harness's thematic proxy and blinded rating sheet instead.

Usage:
    python experiments/reproduce_paper.py [--model MODEL] [--quick]

Requirements:
    - pip install -e ".[research]"
    - A GPU helps; the default model also runs on CPU
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from machine_poi.steerer import QuranSteerer


def run_qualitative_comparison(steerer: QuranSteerer, prompts: list[str]) -> dict:
    """
    Run qualitative comparison between steered and unsteered outputs.
    Reproduces Section 5.1 of the paper.
    """
    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON (Section 5.1)")
    print("=" * 70)
    
    results = []
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 50)
        
        # Generate unsteered
        unsteered = steerer.generate_unsteered(prompt, max_new_tokens=150)
        print(f"\n🔹 Unsteered:\n{unsteered[:500]}...")
        
        # Generate steered
        steered = steerer.generate(prompt, max_new_tokens=150)
        print(f"\n🔸 Steered (Quran Persona):\n{steered[:500]}...")
        
        results.append({
            "prompt": prompt,
            "unsteered": unsteered,
            "steered": steered,
        })
    
    return results


def run_mra_comparison(steerer: QuranSteerer, prompt: str) -> dict:
    """
    Run MRA (Multi-Resolution Analysis) mode comparison.
    Tests domain bridging functionality.
    """
    print("\n" + "=" * 70)
    print("MRA MODE WITH DOMAIN BRIDGING")
    print("=" * 70)
    
    print(f"\n📝 Prompt: {prompt}")
    print("-" * 50)
    
    # Generate with MRA mode
    mra_output = steerer.generate(
        prompt, 
        max_new_tokens=200, 
        mra_mode=True,
        use_domain_bridges=True
    )
    print(f"\n🔸 MRA Output:\n{mra_output}")
    
    return {"prompt": prompt, "mra_output": mra_output}


def run_dose_demo(steerer: QuranSteerer, prompt: str, dose_ratios: list[float]) -> list[dict]:
    """Print outputs across dose ratios (section 5.3); a demonstration, not a score."""
    print("\n" + "=" * 70)
    print("DOSE DEMONSTRATION (Section 5.3)")
    print("=" * 70)

    print(f"\n📝 Test Prompt: {prompt}")

    results = []
    previous = steerer.config.dose_ratio
    try:
        for ratio in dose_ratios:
            steerer.set_dose_ratio(ratio)
            output = steerer.generate(prompt, max_new_tokens=100)
            result = {
                "dose_ratio": ratio,
                "output_length": len(output),
                "output_preview": output[:200] + "..." if len(output) > 200 else output,
            }
            results.append(result)
            print(f"\n   dose ratio {ratio}:")
            print(f"   Output: {result['output_preview']}")
    finally:
        if previous is not None:
            steerer.set_dose_ratio(previous)

    return results


def main():
    parser = argparse.ArgumentParser(description="Machine-POI steering demonstrations")
    parser.add_argument("--model", default="deepseek-r1-1.5b", help="LLM model to use")
    parser.add_argument("--embedding", default="paraphrase-minilm", help="Embedding model to use")
    parser.add_argument("--quick", action="store_true", help="Run quick version with fewer prompts")
    parser.add_argument("--section", choices=["all", "5.1", "5.2", "5.3", "mra"], 
                        default="all", help="Which section to run (5.2 is retired)")
    args = parser.parse_args()

    if args.section == "5.2":
        print("Section 5.2 (keyword counting) is retired. Run "
              "experiments/steering_eval.py; see docs/evaluation.md.")
        return

    print("=" * 70)
    print("MACHINE-POI STEERING DEMONSTRATION")
    print(f"Model: {args.model} | Embedding: {args.embedding}")
    print("=" * 70)
    
    # Initialize steerer
    print("\n🔄 Loading models...")
    steerer = QuranSteerer(
        llm_model=args.model,
        embedding_model=args.embedding,
    )
    steerer.load_models()
    
    # Prepare Quran Persona steering
    print("\n🔄 Preparing Quran Persona steering vectors...")
    steerer.prepare_quran_persona(cache_dir="vectors")
    
    # Define test prompts
    qualitative_prompts = [
        "What is the meaning of life?",
        "How should I deal with a bug in my code?",
    ]
    
    if args.quick:
        qualitative_prompts = qualitative_prompts[:1]
    
    # Run experiments based on selection
    if args.section in ["all", "5.1"]:
        run_qualitative_comparison(steerer, qualitative_prompts)
    
    if args.section in ["all", "mra"]:
        run_mra_comparison(steerer, "How should I deal with a bug in my code?")
    
    if args.section in ["all", "5.3"]:
        run_dose_demo(steerer, "What is patience?", [0.02, 0.05, 0.1, 0.2])
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
