import itertools
import subprocess

models = ["Qwen/Qwen2.5-3B-Instruct"]
lora_ranks = [64]
seq_lengths = [1024]
# learning_rates = [1e-7, 2.5e-7, 5e-7, 1e-6, 2.5e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4]
# grad_clips = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# lr_schedules = ["constant", "constant_with_warmup"]
learning_rates = [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2]
grad_clips = [0.005, 0.01, 0.05]
lr_schedules = ["constant", "constant_with_warmup"]

i = 0

script_template = '''#!/bin/bash
conda activate unsloth
cd ~/unsloth
'''

script = script_template.format(name=i//4)

for model, lora_rank, seq_length, learning_rate, grad_clip, lr_schedule in \
  itertools.product(models, lora_ranks, seq_lengths, learning_rates, grad_clips, lr_schedules):

  name = f"lr {learning_rate:.1e} grad clip {grad_clip} {lr_schedule}"
  command = f"\npython experiments/train_grpo.py -m \"{model}\" -d {i % 4} --lora_rank {lora_rank} --seq_length {seq_length} --learning_rate {learning_rate} --grad_clip {grad_clip} --lr_schedule \"{lr_schedule}\" --output_dir \"{name}\" &"

  script += command

  if (i + 1) % 4 == 0:
    script += "\nwait"
    # launch the current script,
    with open(f"scripts/tmp_{i//4}.sh", "w") as f:
      f.write(script)

    script = script_template.format(name=(i+1)//4)

  i += 1

print(f'Total runs: {i}')
nodes = 15

for j in range(0, i//4, nodes):
  
  main_script = f'''#!/bin/bash
#PBS -N sweep_{j//nodes}
#PBS -l filesystems=home:eagle
#PBS -l select={min(nodes, i//4 - j)}
#PBS -l walltime=03:00:00
#PBS -q prod
#PBS -A DemocAI
''' + '''
NODES=($(sort -u $PBS_NODEFILE))
NNODES=$(wc -l < $PBS_NODEFILE)

for i in $(seq 0 $(($NNODES - 1)));
do
    node=${NODES[$i]}''' + f'''
    script=~/unsloth/scripts/tmp_$((i + {j})).sh
    echo "Running $script on $node"
    ssh "$node" "bash $script" &
done

wait
'''

  with open(f"scripts/main_{j}.sh", "w") as f:
    f.write(main_script)

  subprocess.run(f"qsub scripts/main_{j}.sh", shell=True)

# qselect -u vatsalb | xargs qdel