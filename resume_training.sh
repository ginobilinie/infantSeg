#!/usr/bin/env sh

caffe train \
    --solver=infant_solver.prototxt \
    --snapshot=infant_fcn_iter_16000.solverstate
