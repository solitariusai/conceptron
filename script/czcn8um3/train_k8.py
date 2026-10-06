import argparse
from typing import Any

import grain.python as grain
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from jax.sharding import AxisType
from taktiny import nn
from taktiny.data import BatchMap, DataLoader, Pack, train_validation_split
from taktiny.trainer import DatasetConfig, Trainer, TrainingConfig
from taktiny.utils import map_logical_axis_names

from conceptron._parts import ConceptronCache
from conceptron.exp.czcn8um3 import ControlConfig, Exp_czcn8um3
from conceptron.proc import TokenizerExp


@jax.jit
def forward(model, ids, mask, cache) -> jax.Array:
    logits = model(ids, mask, cache=cache)
    return logits

def generate(model: Exp_czcn8um3, tokenizer: Any, prompt: str, max_new_tokens: int, cache: ConceptronCache):
    print(prompt, end='', flush=True)
    ids = jnp.asarray(tokenizer.encode(prompt, return_tensors='np'))
    mask = jnp.tril(jnp.ones((ids.shape[1], cache.cache_length), dtype=jnp.bool_))
    for _ in range(max_new_tokens):
        logits: jax.Array = forward(model, ids, mask, cache)
        mask = None
        max_probs_id = logits[:, -1].argmax(-1, keepdims=True)
        ids = max_probs_id
        decode = tokenizer.decode(ids[0])
        prompt += decode
        print(decode, end='', flush=True)

    print()

def process_dataset(repo: str, max_len: int, batch_size: int, workers: int, val_rows: int):
    ds = load_dataset(repo, split='train').select_columns('text')
    train, val = train_validation_split(ds, 0.1)
    def tokenize(rows):
        rows = jax.tree.map(lambda *r: list(r), *rows)
        return {'input_ids': tok.encode(rows['text'])}  # ty: ignore[unresolved-attribute]

    train_loader = DataLoader(
        train, 
        operations=[
            BatchMap(tokenize, batch_size=1024, drop_remainder=True),
            Pack(max_len, keys='input_ids', position_key='position_ids', drop_remainder=True)
        ],
        worker_buffer_size=2,
        worker_count=workers,
        read_options=grain.ReadOptions(
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        axis_names=('batch', 'sequence'),
        batch_size=batch_size,
        drop_remainder=True,
    )
    val_loader = DataLoader(
        val[:val_rows], 
        operations=[
            BatchMap(tokenize, batch_size=val_rows // max(workers, 1), drop_remainder=True),
            Pack(max_len, keys='input_ids', position_key='position_ids', drop_remainder=True)
        ],
        worker_buffer_size=2,
        worker_count=workers,
        read_options=grain.ReadOptions(
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        axis_names=('batch', 'sequence'),
        batch_size=batch_size,
        drop_remainder=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data-repo', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-rows', type=int, default=1)
    parser.add_argument('--max-seq-len', type=int, default=128)

    # train
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--loss-chunk-size', type=int, default=128)

    # other
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--not-save', action='store_false', default=True) # debug

    args = parser.parse_args()
    if args.loss_chunk_size < 1:
        parser.error('--loss-chunk-size must be positive')

    mesh = jax.make_mesh(
        (1, jax.device_count()), 
        ('model', 'data'), 
        (AxisType.Auto, AxisType.Auto)
    )
    jax.set_mesh(mesh)
    map_logical_axis_names({
        'vocab': None,
        'hidden': 'model',
        'num_heads': None,
        'head_dim': None,
        'intermediate': None,
        'batch': 'data',
        'sequence': None,
    })

    train_loader, val_loader = process_dataset(
        args.data_repo, 
        args.max_seq_len, 
        args.batch_size, 
        args.workers, 
        args.eval_rows
    )

    config = ControlConfig()
    tok = TokenizerExp()

    def loss_fn(model, batch):
        input_ids = batch['input_ids']

        def pass_loss_fn(x, lm_head_ref):
            # Chunk token positions across the whole batch, not each sequence.
            hidden = x[:, :-1].reshape(-1, x.shape[-1])
            targets = input_ids[:, 1:].reshape(-1)
            num_tokens = targets.shape[0]
            if num_tokens == 0:
                raise ValueError('next-token loss requires at least two tokens per sequence')
            chunk_size = min(args.loss_chunk_size, num_tokens)
            padding = (-num_tokens) % chunk_size
            hidden = jnp.pad(hidden, ((0, padding), (0, 0)))
            targets = jnp.pad(targets, ((0, padding),))
            valid = jnp.arange(num_tokens + padding) < num_tokens
            lm_head = lm_head_ref[...]

            def chunk_loss(hidden_chunk, target_chunk, valid_chunk):
                logits = jnp.dot(hidden_chunk, lm_head).astype(jnp.float32)
                losses = optax.softmax_cross_entropy_with_integer_labels(logits, target_chunk)
                return jnp.where(valid_chunk, losses, 0.0).sum()

            remat_chunk_loss = jax.checkpoint(chunk_loss, prevent_cse=False)

            def step(total, inputs):
                return total + remat_chunk_loss(*inputs), None

            total, _ = jax.lax.scan(
                step,
                jnp.asarray(0.0, dtype=jnp.float32),
                (
                    hidden.reshape(-1, chunk_size, x.shape[-1]),
                    targets.reshape(-1, chunk_size),
                    valid.reshape(-1, chunk_size),
                ),
            )
            return total / num_tokens

        position_ids = batch['position_ids']
        segment_ids = jnp.cumsum(position_ids == 0, -1) - 1
        mask = jnp.tril(segment_ids[..., :, None] == segment_ids[..., None, :])[:, None, ...]
        return model(input_ids, mask, position_ids, loss_fn=pass_loss_fn)

    schedule = optax.cosine_decay_schedule(args.lr, args.max_steps)
    optimizer = optax.adamw(schedule, weight_decay=args.wd)

    model = Exp_czcn8um3(config, rngs=nn.Rngs(0), k=8)
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=args.max_steps,
            schedule=schedule,
            optimizer=optimizer,
            eval_strategy='steps' if args.eval else 'no',
            eval_steps=args.max_steps // 4 if args.max_steps > 10 else args.max_steps,
            output_dir=f'{args.out_dir}-czcn8um3-k8',
            save_at_end=args.not_save,
            log_interval=args.log_interval,
        ),
        DatasetConfig(
            train_loader,
            val_loader
        ),
        loss_fn=loss_fn,
    )
    
    name = 'Exp_czcn8um3-k8'
    print('=' * 20 + f'Start Training {name}' + '=' * 20)
    trainer.train()
    print()

    print('=' * 20 + f'w in {name}' + '=' * 20)
    print(model.w.value)

    print('=' * 20 + f'Generating With {name}' + '=' * 20)
    cache = ConceptronCache(config, 1, 512)
    generate(model, tok, prompt='Hello, ', max_new_tokens=256, cache=cache)
    print('=' * 60)
    cache = ConceptronCache(config, 1, 512)
    generate(model, tok, prompt='AI is', max_new_tokens=256, cache=cache)
    print('=' * 60)
    cache = ConceptronCache(config, 1, 512)
    generate(model, tok, prompt='Mathematics', max_new_tokens=256, cache=cache)
    print('=' * 60)
    cache = ConceptronCache(config, 1, 512)
    generate(model, tok, prompt='Neural Network', max_new_tokens=256, cache=cache)
    print('=' * 60)
    cache = ConceptronCache(config, 1, 512)
    generate(model, tok, prompt='If I can talk like other Language Model I want to say', max_new_tokens=256, cache=cache)
    print('=' * 60)
