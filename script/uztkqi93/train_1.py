from typing import Any

import grain.python as grain
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from taktiny import nn
from taktiny.data import BatchMap, DataLoader, Pack, train_validation_split
from taktiny.trainer import DatasetConfig, Trainer, TrainingConfig
from taktiny.utils import map_logical_axis_names

from conceptron._parts import ConceptronCache
from conceptron.exp.uztkqi93 import ControlConfig, Exp_uztkqi93
from conceptron.proc import TokenizerExp


@jax.jit
def forward(model, ids, mask, cache) -> jax.Array:
    logits = model(ids, mask, cache=cache)
    return logits

def generate(model: Exp_uztkqi93, tokenizer: Any, prompt: str, max_new_tokens: int, cache: ConceptronCache):
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
        return {'input_ids': tok.encode(rows['text'])} # ty: ignore[unresolved-attribute]

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
        axis_names=('batch', 'seq'),
        batch_size=batch_size,
        drop_remainder=True,
    )
    val_loader = DataLoader(
        val[:val_rows], 
        operations=[
            BatchMap(tokenize, batch_size=val_rows // workers, drop_remainder=True),
            Pack(max_len, keys='input_ids', position_key='position_ids', drop_remainder=True)
        ],
        worker_buffer_size=2,
        worker_count=workers,
        read_options=grain.ReadOptions(
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        axis_names=('batch', 'seq'),
        batch_size=batch_size,
        drop_remainder=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import argparse

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

    # other
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--not-save', action='store_false', default=True) # debug

    args = parser.parse_args()

    mesh = jax.make_mesh((jax.device_count(), 1), ('model', 'data'))
    jax.set_mesh(mesh)
    map_logical_axis_names({
        
    })

    train_loader, val_loader = process_dataset(
        args.data_repo, args.max_seq_len, args.batch_size, args.workers, args.eval_rows
    )

    config = ControlConfig()
    tok = TokenizerExp()
    
    def loss_fn_base(model, batch):
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        segment_ids = jnp.cumsum(position_ids == 0, -1) - 1
        mask = jnp.tril(segment_ids[..., :, None] == segment_ids[..., None, :])[:, None, ...]
        logits = model(input_ids, mask, position_ids)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1],
            input_ids[:, 1:]
        )
        return loss.mean()

    def loss_fn_exp(model, batch):
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        segment_ids = jnp.cumsum(position_ids == 0, -1) - 1
        mask = jnp.tril(segment_ids[..., :, None] == segment_ids[..., None, :])[:, None, ...]
        logits = model(input_ids, mask, position_ids, base=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1],
            input_ids[:, 1:]
        )
        return loss.mean()

    schedule = optax.cosine_decay_schedule(args.lr, args.max_steps)
    optimizer = optax.adamw(schedule, weight_decay=args.wd)

    base_model = Exp_uztkqi93(config, rngs=nn.Rngs(0))
    trainer = Trainer(
        base_model,
        TrainingConfig(
            max_steps=args.max_steps,
            schedule=schedule,
            optimizer=optimizer,
            eval_strategy='steps' if args.eval else 'no',
            eval_steps=args.max_steps // 4 if args.max_steps > 10 else args.max_steps,
            output_dir=f'{args.out_dir}-base',
            save_at_end=args.not_save,
            log_interval=args.log_interval,
        ),
        DatasetConfig(
            train_loader,
            val_loader
        ),
        loss_fn=loss_fn_base,
    )
    
    print('=' * 20 + 'Start Training Base Model' + '=' * 20)
    trainer.train()
    print()

    print('=' * 20 + 'Generating With Base Model' + '=' * 20)
    cache = ConceptronCache(config, 1, 256)
    generate(base_model, tok, prompt='Hello, ', max_new_tokens=64, cache=cache)
    del base_model
    print('=' * 60)

    exp_model = Exp_uztkqi93(config, rngs=nn.Rngs(0))
    trainer = Trainer(
        exp_model,
        TrainingConfig(
            max_steps=args.max_steps,
            schedule=schedule,
            optimizer=optimizer,
            eval_strategy='steps' if args.eval else 'no',
            eval_steps=args.max_steps // 4 if args.max_steps > 10 else args.max_steps,
            output_dir=f'{args.out_dir}-exp',
            save_at_end=args.not_save,
            log_interval=args.log_interval,
        ),
        DatasetConfig(
            train_loader,
            val_loader
        ),
        loss_fn=loss_fn_exp,
    )

    print('=' * 20 + 'Start Training Exp Model' + '=' * 20)
    trainer.train()
    print()

    print('=' * 20 + 'Generating With Exp Model' + '=' * 20)
    cache = ConceptronCache(config, 1, 256)
    generate(exp_model, tok, prompt='Hello, ', max_new_tokens=64, cache=cache)
    print()
