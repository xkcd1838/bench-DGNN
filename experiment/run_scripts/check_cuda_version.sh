singularity exec --nv ../../container.sif python -c "import torch; print(torch.version.cuda); print('cuda available', torch.cuda.is_available())"
