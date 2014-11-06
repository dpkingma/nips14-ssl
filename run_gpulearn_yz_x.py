import gpulearn_yz_x
import sys

if sys.argv[1] == 'svhn':
    n_hidden = [500,500]
    if len(sys.argv) == 4:
        n_hidden = [int(sys.argv[2])]*int(sys.argv[3])
    gpulearn_yz_x.main(dataset='svhn', n_z=300, n_hidden=n_hidden, seed=0, gfx=True)

elif sys.argv[1] == 'mnist':
    n_hidden = (500,500)
    if len(sys.argv) >= 4:
        n_hidden = [int(sys.argv[2])]*int(sys.argv[3])
    n_z = 50
    if len(sys.argv) >= 5:
        n_z = int(sys.argv[4])
    gpulearn_yz_x.main(dataset='mnist', n_z=n_z, n_hidden=n_hidden, seed=0, gfx=True)
raise Exception("Unknown dataset")
