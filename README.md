# WARP-SPEED

A bipartite factorisation machine with a modified WARP loss [1] for hybrid recommendation, written in Cython.  Once used at Lateral for recommendation at scale.  No longer maintained.

## Not exactly WARP!

The implementation deviates from the standard in that it uses a sigmoid activation, a multiplicative margin, and a log-likelihood style updates.  These modifications were justified by (unpublished, internal) performance improvements on customer datasets.

In addition, we use a modification of Adagrad that works at the feature _vector_ level, instead of at the level of coordinates.  This greatly reduces the number of Adagrad parameters and moreover makes the optimisation [independent of the choice of coordinate axes](http://building-babylon.net/2016/10/05/adagrad-evolution-depends-on-the-choice-of-basis/).

## Requirements

+ Python >= 3.5
+ Cython == 0.25.2

See also `requirements.txt`.

## Installation

```
./install
```

## Usage

TODO add example notebook

## References

[1] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
