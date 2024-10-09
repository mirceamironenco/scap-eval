from typing import Final

from scap.tasks import register_task, task_dataset_decorator
from scap.tasks.recall._mqar import DisjointSetCfg, MQARCfg

RECALL_TASK: Final = "recall"

register_task(name=RECALL_TASK)
recall_dataset = task_dataset_decorator(task_name=RECALL_TASK)


@recall_dataset()
def disjoint_sets(num_examples: int) -> DisjointSetCfg:
    return DisjointSetCfg(
        num_examples=num_examples, vocab_size=2048, short_length=4, long_length=16
    )


@recall_dataset()
def mqar(num_examples: int) -> MQARCfg:
    return MQARCfg(num_examples=num_examples)


@recall_dataset()
def mqar_sql64_kv4(num_examples: int) -> MQARCfg:
    return MQARCfg(num_examples=num_examples, input_seq_len=64, num_kv_pairs=4)


@recall_dataset()
def mqar_sql128_kv8(num_examples: int) -> MQARCfg:
    return MQARCfg(num_examples=num_examples, input_seq_len=128, num_kv_pairs=8)


@recall_dataset()
def mqar_sql256_kv16(num_examples: int) -> MQARCfg:
    return MQARCfg(num_examples=num_examples, input_seq_len=256, num_kv_pairs=16)


@recall_dataset()
def mqar_sql512_kv64(num_examples: int) -> MQARCfg:
    return MQARCfg(num_examples=num_examples, input_seq_len=512, num_kv_pairs=64)
