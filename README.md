# microgpt-test

Demo repository showcasing [MicroGPT](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/2d7d2d2c9b32b27282bb572ca72224f5c3d5c120/microgpt.py) - a minimal GPT implementation by Andrej Karpathy.

## Files

- **microgpt.py** - The complete GPT implementation (single file, no dependencies)
- **microgpt-demo.md** - Interactive demo document created with [showboat](https://www.npmjs.com/package/showboat)

## Running the Demo

### View the Demo Document

Simply open `microgpt-demo.md` in any markdown viewer to see:
- Overview of MicroGPT's architecture
- Training run with captured output
- Generated name samples
- Analysis and conclusions

### Verify the Demo

To re-run all code blocks and verify the outputs still match:

```bash
npx showboat verify microgpt-demo.md
```

### Extract Commands

To see the sequence of commands used to create the demo:

```bash
npx showboat extract microgpt-demo.md
```

### Run MicroGPT Directly

```bash
python3 microgpt.py --help
```

Example with custom parameters:

```bash
python3 microgpt.py --n_embd 16 --n_layer 1 --n_head 4 --block_size 8 --num_steps 100
```

## About Showboat

[Showboat](https://www.npmjs.com/package/showboat) is a tool for creating executable demo documents that mix:
- Commentary and explanations
- Executable code blocks
- Captured output

These documents serve as both readable documentation and reproducible proof of work.

## About MicroGPT

MicroGPT is an educational implementation of the GPT architecture featuring:
- Pure Python with no external dependencies
- Custom autograd engine for gradient computation
- Full transformer architecture (attention, normalization, feedforward)
- Training and inference in ~250 lines of code

Perfect for learning how modern language models work under the hood!