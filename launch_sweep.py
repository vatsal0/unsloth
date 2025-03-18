import itertools
import subprocess

models = ["Qwen/Qwen2.5-3B-Instruct"]
lora_ranks = [64]
seq_lengths = [1024]
learning_rates = [1e-7, 2.5e-7, 5e-7, 1e-6, 2.5e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4]
grad_clips = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
lr_schedules = ["constant", "constant_with_warmup"]

i = 0

script_template = '''#!/bin/bash
#PBS -N {name}
#PBS -l filesystems=home:eagle
#PBS -l select=1
#PBS -l walltime=05:00:00
#PBS -q preemptable
#PBS -A DemocAI

conda activate unsloth
cd ~/unsloth
'''

script = script_template.format(name=i//4)

for model, lora_rank, seq_length, learning_rate, grad_clip, lr_schedule in \
  itertools.product(models, lora_ranks, seq_lengths, learning_rates, grad_clips, lr_schedules):

  name = f"lr {learning_rate:0.8f} grad clip {grad_clip} {lr_schedule}"
  command = f"\npython experiments/train_grpo.py -m \"{model}\" -d {i % 4} --lora_rank {lora_rank} --seq_length {seq_length} --learning_rate {learning_rate} --grad_clip {grad_clip} --lr_schedule \"{lr_schedule}\" --output_dir \"{name}\" &"

  script += command

  if (i + 1) % 4 == 0:
    script += "\nsleep 18000"
    # launch the current script,
    with open(f"scripts/tmp_{i//4}.sh", "w") as f:
      f.write(script)

    subprocess.run(f"qsub scripts/tmp_{i//4}.sh", shell=True)

    script = script_template.format(name=(i+1)//4)

  i += 1

# qselect -u vatsalb | xargs qdel