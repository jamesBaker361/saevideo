
sbatch -J sana --err=slurm_chip/test/synthetic_sana.py_gpu.err --out=slurm_chip/test/synthetic_sana.py_gpu.out runpygpu_chip_heavy.sh synthetic_sana.py

sbatch -J sd3 --err=slurm_chip/test/synthetic_sd3.py_gpu.err --out=slurm_chip/test/synthetic_sd3.py_gpu.out runpygpu_chip_heavy.sh synthetic_sd3.py 