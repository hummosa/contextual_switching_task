import os
from subprocess import call
import subprocess as sp
from time import sleep
import sys

black_surface = False
sbatch_lines =[
        "# Read more about SSH config files: https://linux.die.net/man/5/ssh_config",
        "Host gateway",
        "\tHostName     openmind7.mit.edu",
        "\tUser         ahummos",
        "Host openmind",
        "\tHostName     openmind7.mit.edu",
        "\tForwardAgent yes",
        "\tUser         ahummos",
        "Host *",
            "\tStrictHostKeyChecking no",
        ]

expVars = range(2, 120)
for node_num in expVars:
    sbatch_lines+= [f'Host node{node_num:03}',
    f'\tHostName     node{node_num:03}',
    "\tUser         ahummos",]
    if black_surface:
        sbatch_lines += ['\tProxyCommand ssh -t -W %h:%p gateway']
    else:
        sbatch_lines += ['\tProxyJump    gateway\n']


print('writing ssh_config.file..')
fsh = open('.\ssh_config.file', 'w')
fsh.write("\n".join(sbatch_lines))
# fsh.writelines(["\n", command_line])
fsh.close()