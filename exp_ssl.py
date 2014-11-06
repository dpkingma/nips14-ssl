import learn_yz_x_ss
import sys
learn_yz_x_ss.main(n_passes=3000, n_labeled=int(sys.argv[1]), dataset='mnist_2layer', n_z=50, n_hidden=tuple([int(sys.argv[2])]*int(sys.argv[3])), seed=int(sys.argv[4]), alpha=0.1, comment='')
